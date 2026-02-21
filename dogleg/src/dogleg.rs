use crate::dogleg::common::gtol_calc;
use crate::Error;
use crate::LeastSquaresProblem;
use crate::MagicConst;
use crate::TerminationFailure;
// use assert2::debug_assert;
use dogleg_matx::Addx;
use dogleg_matx::ColEnormsx;
use dogleg_matx::DiagLeftMulx;
use dogleg_matx::DiagRightMulx;
use dogleg_matx::ElementwiseMaxx;
use dogleg_matx::ElementwiseReplaceLeqx;
use dogleg_matx::Invert;
use dogleg_matx::MaxScaledDivx;
use dogleg_matx::{Colx, TrMatVecMulx};
use num_traits::Float;
use num_traits::FromPrimitive;
use std::num::NonZero;

mod common;
mod qr_impl;
mod reset_guard;
mod svd_impl;

pub mod report;
pub use common::DoglegStep;
pub use common::DoglegStepSolver;
pub use report::MinimizationReport;
pub use report::TerminationReason;
pub use svd_impl::SvdStepSolver;

/// like debug_assert_eq, but doesn't require lhs, rhs to implement the Debug
/// trait.
macro_rules! debug_assert_eq2 {
    ($lhs:expr, $rhs:expr $(,)?) => {
        #[cfg(debug_assertions)]
        {
            if $lhs != $rhs {
                panic!(
                    "Debug assertion failed: {}=={}",
                    stringify!($lhs),
                    stringify!($rhs)
                )
            }
        }
    };
}

/// utility macro that helps us return our error inline when the
/// returned type of an expression is an optional. The challenge is
/// that we need to bundle our problem and return it together with
/// the error and if we use the monadic interfaces of Option<T>
/// we'll get into trouble with the borrow checker.
///
/// On none, returns the error with the given termination criterion
/// and bundled with the problem.
macro_rules! try_opt {
    // assumes the $problem is the actual problem. Use the
    // second syntax below to specify the problem
    ($problem:ident . $function:ident ($($tokens:tt)*), on_none
         = $failure:expr) => {
        match $problem. $function ($($tokens)*) {
            Some(val) => val,
            None => {
                return Err($crate::Error {
                    problem: $problem,
                    failure : $failure
                });
            }
        }
    };

    // for calls like jacobian.mul_tr(&residuals)
    // also if there's a RAII drop guard, this will drop it before return, if given
    ($expr:expr, on_none = $failure:expr, problem = $problem:ident $(,guard = $guard:ident)? $(,)?) => {
        match $expr {
            Some(val) => val,
            None => {
                $(drop($guard);)?
                return Err($crate::Error {
                    problem: $problem,
                    failure : $failure
                });
            }
        }
    };
}

/// a try macro for results that can be
macro_rules! try2 {
    ($expr:expr, problem = $problem:ident) => {
        match $expr {
            Ok(val) => val,
            Err(failure) => {
                return Err($crate::Error {
                    problem: $problem,
                    failure,
                });
            }
        }
    };
}

/// Powell's Dogleg minimization algorithm. The behaviour of the algorithm
/// can be controlled by setting various parameters.
///
/// Note, that the Dogleg algorithm requires the Jacobian of the problem
/// to be a full-rank matrix at for every position that is possibly evaluated.
/// This is not a limitation of the implementation, but a fundamental assumption
/// in the algorithm.
///
/// # References
///
/// Nocedal & Wright: Numerical Optimization, 2nd ed, p 73-76, 95-97, 245-247
#[derive(Debug, PartialEq, Clone)]
pub struct Dogleg<T> {
    /// Relative error criterion on the residual values.
    /// See section 2.3 in the MINPACK user guide: https://cds.cern.ch/record/126569/files/CM-P00068642.pdf
    ftol: T,
    /// Attempt to guarantee that x is in the vicinity of a true solution, by
    /// estimating the distance of the true and current x by
    /// See section 2.3 in the MINPACK user guide: https://cds.cern.ch/record/126569/files/CM-P00068642.pdf
    xtol: T,
    /// Value for checking the orthogonality between residuals and Jacobian.
    /// It's a more clever (scale-invariant) way of checking whether the gradient
    /// of the problem is zero.
    /// See section 2.3 in the MINPACK user guide: https://cds.cern.ch/record/126569/files/CM-P00068642.pdf
    gtol: T,
    /// see `lmder` function in the minpack code (https://github.com/fortran-lang/minpack/blob/main/src/minpack.f90)
    /// >> a positive input variable used in determining the
    /// >> initial step bound. this bound is set to the product of
    /// >> factor and the euclidean norm of diag*x if nonzero, or else
    /// >> to factor itself. in most cases factor should lie in the
    /// >> interval (.1,100.).100. is a generally recommended value
    factor: T,
    /// Whether to apply diagonal scaling internally. For this, see
    /// Nocedal & Wright, p 95-97.
    use_elliptical_parameter_scaling: bool,
    /// whether to apply Jacobi-scaling, which is different from the elliptical
    /// parameter scaling. This is something that CERES solver does,
    /// see e.g.: https://github.com/ceres-solver/ceres-solver/blob/a2bab5af5131d52a756b1fa7b7cff83821541449/internal/ceres/trust_region_minimizer.cc#L263.
    /// This is also a type of diagonal scaling, but it's calculated differently
    /// from the elliptical parameter scaling. This is only calculated once,
    /// based on the initial jacobian and acts as a "pre-conditioner" on the
    /// jacobian using, where scaling = diag( (J^T J)^-1). Of course, it
    /// must always be accounted when calculating the unscaled parameters.
    use_jacobi_scaling: bool, 
    /// Used to calculate the maximum number of function evals (a stopping
    /// criterion) based on the problem
    patience: u64,
    /// minimum value for the diagonal scaling matrix. If used, the diagonal
    /// values will be clamped to the min and maximum values.
    min_diagonal : T,
    /// maximum value for the diagonal scaling matrix. If used, the diagonal
    /// values will be clamped to the min and maximum values.
    max_diagonal : T
}

impl<T> Default for Dogleg<T>
where
    T: Float + MagicConst,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Dogleg<T>
where
    T: Float + MagicConst,
{
    /// Create a solver with reasonable default parameters. Consider changing
    /// the parameters if optimization results are unsatisfying.
    pub fn new() -> Self {
        // this logic is taken from the brilliant `levenberg-marquardt` crate
        let user_tol = T::epsilon() * T::THIRTY;
        Self {
            ftol: user_tol,
            xtol: user_tol,
            gtol: user_tol,
            // the min and max diagonal default values are taken from
            // CERES solver, see: https://github.com/ceres-solver/ceres-solver/blob/a2bab5af5131d52a756b1fa7b7cff83821541449/internal/ceres/trust_region_strategy.h#L67
            // but note that the values in the ceres score are applied to the
            // squared norms for clipping, so we have to take the square roots.
            min_diagonal : MagicConst::P001,
            max_diagonal : MagicConst::ONE_E16,
            factor: T::ONE_HUNDRED,
            use_elliptical_parameter_scaling: true,
            patience: 100,
            use_jacobi_scaling: true,
        }
    }

    /// Set `ftol` for the termination criterion for function reduction
    /// according to the same logic as MINPACK:
    ///
    /// > termination occurs when both the actual and predicted relative
    /// > reductions in the sum of squares are at most `ftol`.
    /// > therefore, `ftol` measures the relative error desired
    /// > in the sum of squares.
    ///
    /// Cf. function `lmder` in the [MINPACK implementation](https://github.com/fortran-lang/minpack/blob/main/src/minpack.f90).
    ///
    /// # Panics
    ///
    /// Panics if `ftol < 0`.
    #[must_use]
    pub fn with_ftol(self, ftol: T) -> Self {
        assert!(ftol.is_finite() && ftol >= T::ZERO, "ftol < 0 not allowed");
        Self { ftol, ..self }
    }

    /// Set `xtol` for the termination criterion for consecutive iterates,
    /// according to this logic from the MINPACK implementation:
    ///
    /// > termination occurs when the relative error between two consecutive
    /// > iterates is at most xtol. therefore, xtol measures the
    /// > relative error desired in the approximate solution.
    ///
    /// Cf. function `lmder` in the [MINPACK implementation](https://github.com/fortran-lang/minpack/blob/main/src/minpack.f90).
    /// # Panics
    ///
    /// If `xtol` is negative.
    #[must_use]
    pub fn with_xtol(self, xtol: T) -> Self {
        assert!(xtol.is_finite() && xtol >= T::ZERO, "xtol < 0 not allowed");
        Self { xtol, ..self }
    }

    #[must_use]
    /// Set `gtol` for the termination criterion the gradient
    /// according to this logic from the MINPACK implementation:
    ///
    /// > termination occurs when the cosine of the angle between
    /// > \[the residuals\] and any column of the jacobian is at most `gtol` in absolute
    /// > value. therefore, gtol measures the orthogonality
    /// > desired between the \[residual\] vector and the columns
    /// > of the jacobian.
    ///
    /// Cf. function `lmder` in the [MINPACK implementation](https://github.com/fortran-lang/minpack/blob/main/src/minpack.f90).
    /// # Panics
    ///
    /// If `gtol` is negative.
    pub fn with_gtol(self, gtol: T) -> Self {
        assert!(gtol.is_finite() && gtol >= T::ZERO, "gtol < 0 not allowed");
        Self { gtol, ..self }
    }

    #[must_use]
    /// Shortcut to set `ftol = xtol = tol` and `gtol = 0`, which is what the
    /// high level driver function `lmder1` in MINPACK does.
    ///
    /// Cf. function `lmder` in the [MINPACK implementation](https://github.com/fortran-lang/minpack/blob/main/src/minpack.f90).
    ///
    ///
    pub fn with_tol(self, tol: T) -> Self {
        assert!(tol.is_finite() && tol > T::ZERO, "tol < 0 not allowed");
        Self {
            ftol: tol,
            xtol: tol,
            gtol: T::ZERO,
            ..self
        }
    }

    #[must_use]
    /// Used to set the initial radius of the trust region, according to this
    /// logic from the MINPACK implementation:
    ///
    /// > a positive input variable used in determining the
    /// > initial step bound. this bound is set to the product of
    /// > factor and the euclidean norm of diag*x if nonzero, or else
    /// > to factor itself. in most cases factor should lie in the
    /// > interval (.1, 100). 100 is a generally recommended value.
    ///
    /// Cf. function `lmder` in the [MINPACK implementation](https://github.com/fortran-lang/minpack/blob/main/src/minpack.f90).
    ///
    /// # Panics
    ///
    /// If stepbound is negative
    pub fn with_stepbound(self, stepbound: T) -> Self {
        assert!(
            stepbound.is_finite() && stepbound > T::ZERO,
            "step bound < 0 not allowed"
        );
        Self {
            factor: stepbound,
            ..self
        }
    }

    #[must_use]
    /// This sets the maximum number of function evaluations to
    /// `patience * (n+1)`, where n is the number of parameters.
    pub fn with_patience(self, patience: NonZero<u64>) -> Self {
        Self {
            patience: patience.get(),
            ..self
        }
    }

    #[must_use]
    #[deprecated = "use with_diagional_scaling(...)"]
    /// Whether to apply diagonal rescaling of variables, see e.g.
    /// Nocedal & Wright p 95-97.
    ///
    /// **Deprectated**, but for better drop-in compatibility
    /// with the `levenberg_marquardt` crate.
    pub fn with_scale_diag(self, use_scaling: bool) -> Self {
        Self { use_elliptical_parameter_scaling: use_scaling, ..self }
    }

    #[must_use]
    /// Whether to apply diagonal rescaling of variables, see e.g.
    /// Nocedal & Wright p 95-97
    pub fn with_diagonal_scaling(self, scale_diag: bool) -> Self {
        Self { use_elliptical_parameter_scaling: scale_diag, ..self }
    }

    #[must_use]
    /// the minimum value to which the diagonal scaling values will be clamped,
    /// which must be nonnegative. Only has an effect if diagonal scaling is
    /// used.
    pub fn with_min_diag(self, min: T) -> Self {
        debug_assert!(min.is_sign_positive() && !min.is_zero());
        Self {min_diagonal : min, ..self}
    }

    #[must_use]
    /// the maximum value to which the diagonal scaling values will be clamped,
    /// which must be nonnegative. Only has an effect if diagonal scaling is
    /// used.
    pub fn with_max_diag(self, max: T) -> Self {
        debug_assert!(max.is_sign_positive() && !max.is_zero());
        Self {max_diagonal: max, ..self}
    }

    /// whether to use jacobian scaling to try and improve the conditioning
    /// of the Jacobian. While this also acts as a form of diagonal scaling,
    /// this is different from the elliptical parameter scaling.
    pub fn with_jacobi_scaling(self, use_scaling: bool) -> Self {
        Self {use_jacobi_scaling : use_scaling, ..self}
    }
}

/// type of J^T r
type GradType<T, P> = <P as LeastSquaresProblem<T>>::Parameters;

/// step type for a particular dogleg problem is equal to the gradient type
/// (see also the dogleg solver)
type StepType<T, P> = GradType<T, P>;

/// column norms type of matrix J
type ColNormsType<T, J> = <J as ColEnormsx<T>>::Output;

/// Type of the diagonal weights for a problem (owned type of the column norms type)
type DiagonalWeightsType<T, P> =
    <ColNormsType<T, <P as LeastSquaresProblem<T>>::Jacobian> as Colx<T>>::Owned;

// use crate::dogleg::svd_impl::SvdStepSolver;
// use crate::LevMarAdapter;
// use nalgebra::constraint::AreMultipliable;
// use nalgebra::constraint::ShapeConstraint;
// use nalgebra::ClosedAddAssign;
// use nalgebra::Const;
// use nalgebra::Dim;
// use nalgebra::DimMin;
// use nalgebra::DimSub;
// use nalgebra::RealField;
// use nalgebra::Scalar;
// use nalgebra::U1;
// use num_traits::ConstOne;
impl<T> Dogleg<T>
where
    T: std::ops::AddAssign,
{
    pub fn minimize<P>(&self, problem: P) -> Result<(P, MinimizationReport<T>), Error<P>>
    where
        T: Float + MagicConst + std::fmt::Debug+
        //DEBUG(geo)
        FromPrimitive,
        P: LeastSquaresProblem<T>,
        P::Residuals: Clone,
        SvdStepSolver<T, <P as LeastSquaresProblem<T>>::Jacobian, P::Residuals, GradType<T, P>>:
            DoglegStepSolver<
                T,
                Jacobian = P::Jacobian,
                //@note(geo-ant) maybe restricting this output here is not smart,
                // but I think in practice it doesn't matter.
                Gradient = P::Parameters,
                Residuals = P::Residuals,
            >,

        // for max func eval calculation
        // for calculating the gradient g = J^T r
        // @note(geo-ant) what this means is: (J^T * r).to_owned() has the
        // same type as P::Parameters::Owned, which should not be a restriction in practice
        // because the dimensions of J^T r and the parameters must be identical
        //
        // @note(geo-ant): maybe restricting the output of
        P::Jacobian: TrMatVecMulx<T, P::Residuals, Output = P::Parameters> + ColEnormsx<T>,
        // GradType<T, P::Jacobian, P::Residuals>: Colx<T, Owned = <P::Parameters as Colx<T>>::Owned>,
        // for the calculation of the gtol criterion
        // for calculating the diagonal weights and replacing them
        // @note(geo-ant) this assumes that they are the same as the gradient
        // type, i.e. the result of J^T *r.
        DiagonalWeightsType<T, P>:
            ElementwiseReplaceLeqx<T> + ElementwiseMaxx<DiagonalWeightsType<T, P>>,
        // for scaling the jacobian
        P::Jacobian: DiagRightMulx<DiagonalWeightsType<T, P>>,
        // for scaling the gradient and the parameters
        StepType<T, P>: DiagLeftMulx<T, DiagonalWeightsType<T, P>>,
        GradType<T, P>: DiagLeftMulx<T, DiagonalWeightsType<T, P>>,
        // // for calculating the new params x' = x + p = p + x
        // P::Parameters: Addx<T, StepType<T, P>>,
        // for applying the scaling to the parameters
        P::Parameters: DiagLeftMulx<T, DiagonalWeightsType<T, P>>,
        // for gtol calculation
        P::Parameters: MaxScaledDivx<T, DiagonalWeightsType<T, P>>,
        // so that we can add parameters to the step (step type = gradient)
        // for calculating the new params x' = x + p = p + x
        GradType<T, P>: Addx<T, P::Parameters>,
    {
        self.minimize_generic::<SvdStepSolver<_, _, _, _>, _>(problem)
    }

    pub fn minimize_generic<S, P>(
        &self,
        mut problem: P,
    ) -> Result<(P, MinimizationReport<T>), Error<P>>
    where
        T: Float + MagicConst + std::fmt::Debug 
        //DEBUG(geo)
        + FromPrimitive,
        P: LeastSquaresProblem<T>,
        P::Residuals: Clone,
        // see below, we require the gradient, i.e. the of J^T r to be the owned type of P.
        // This should barely be a restriction...
        S: DoglegStepSolver<
            T,
            Jacobian = P::Jacobian,
            //@note(geo-ant) maybe restricting this output here is not smart,
            // but I think in practice it doesn't matter.
            Gradient = P::Parameters,
            Residuals = P::Residuals,
        >,
        // for max func eval calculation
        // for calculating the gradient g = J^T r
        // @note(geo-ant) what this means is: (J^T * r).to_owned() has the
        // same type as P::Parameters::Owned, which should not be a restriction in practice
        // because the dimensions of J^T r and the parameters must be identical
        //
        // @note(geo-ant): maybe restricting the output of
        P::Jacobian: TrMatVecMulx<T, P::Residuals, Output = P::Parameters> + ColEnormsx<T>,
        // GradType<T, P::Jacobian, P::Residuals>: Colx<T, Owned = <P::Parameters as Colx<T>>::Owned>,
        // for the calculation of the gtol criterion
        // for calculating the diagonal weights and replacing them
        // @note(geo-ant) this assumes that they are the same as the gradient
        // type, i.e. the result of J^T *r.
        DiagonalWeightsType<T, P>:
            ElementwiseReplaceLeqx<T> + ElementwiseMaxx<DiagonalWeightsType<T, P>>,
        // for scaling the jacobian
        P::Jacobian: DiagRightMulx<DiagonalWeightsType<T, P>>,
        // for scaling the gradient and the parameters
        StepType<T, P>: DiagLeftMulx<T, DiagonalWeightsType<T, P>>,
        GradType<T, P>: DiagLeftMulx<T, DiagonalWeightsType<T, P>>,
        // // for calculating the new params x' = x + p = p + x
        // P::Parameters: Addx<T, StepType<T, P>>,
        // for applying the scaling to the parameters
        P::Parameters: DiagLeftMulx<T, DiagonalWeightsType<T, P>>,
        // for gtol calculation
        P::Parameters: MaxScaledDivx<T, DiagonalWeightsType<T, P>>,
        // so that we can add parameters to the step (step type = gradient)
        // for calculating the new params x' = x + p = p + x
        GradType<T, P>: Addx<T, P::Parameters>,
    {
        // the current parameters of the problems. We have to keep track of
        // them because the problem struct itself could have parameters set
        // that are discarded. Later in the code we also keep track of the
        // residuals and the jacobian for those parameters. That frees us
        // from having to "fork" the model.
        let mut params = problem.params();
        // @todo(geo-ant) maybe refactor this at a later date, but for the
        // intended use of this library, the number of parameters being near
        // the u64 limit is completely unreasonable, hence panicking here is
        // completely fine. If you have that many parameters, you probably
        // shouldn't be using this algorithm anyway...
        // @note(geo-ant) this whole overflow checking is very pedantic,
        // but what the heck...
        let maybe_dim = params.dim();
        let (n_plus_one, overflow) = try_opt!(
            maybe_dim,
            on_none = TerminationFailure::DimOutsideU64Bounds,
            problem = problem
        )
        .overflowing_add(1);
        let (max_func_evals, overflow2) = self.patience.overflowing_mul(n_plus_one);
        if overflow || overflow2 {
            panic!("too many parameters for dogleg solver");
        }

        // actual number of function evaluations
        let mut nfunc_evals: u64 = 0;

        // @todo(geo-ant)
        // the trust region radius. In the first iteration of the algorithm,
        // this will be set to something useful, so initializing with factor
        // zero here is fine. But we can only calculate this after having
        // calculated the jacobian (and thus the scaling matrix) and pulling
        // this outside makes the loop diverge from the MINPACK implementation,
        // which I'm sticking to for now.
        let mut delta = T::zero();

        // some special case handling for the first loop iteration
        let mut is_first_iteration = true;

        let mut diagonal_weights = None;

        let mut residuals = try_opt!(
            problem.residuals(),
            on_none = TerminationFailure::ResidualEval
        );
        nfunc_evals += 1;
        let mut rnorm = residuals.enorm();
        // f = 1/2 ||r||^2
        let mut objective_function = T::P5 * rnorm.powi(2);

        let mut jacobi_scaling = None;

        // outer loop
        'outer: loop {
            // calculate residuals and jacobian
            if nfunc_evals >= max_func_evals {
                return Err(Error {
                    problem,
                    failure: TerminationFailure::LostPatience,
                });
            }

            if rnorm.is_zero() {
                return Ok((
                    problem,
                    MinimizationReport {
                        termination: TerminationReason::ResidualsZero,
                        number_of_evaluations: nfunc_evals,
                        objective_function,
                    },
                ));
            }

            let mut jacobian = try_opt!(
                problem.jacobian(),
                on_none = TerminationFailure::JacobianEval
            );

            if is_first_iteration && self.use_jacobi_scaling {
                // this takes jacobi scaling from ceres. This is only calculated
                // ONCE from the initial jacobian to try to make it better
                // conditioned. If jacobi_scaling is used, then this is
                // always applied as an extra diagonal scaling
                // see e.g.: https://github.com/ceres-solver/ceres-solver/blob/a2bab5af5131d52a756b1fa7b7cff83821541449/internal/ceres/trust_region_minimizer.cc#L263.
                jacobi_scaling = Some(jacobian.damped_inverse_column_enorms());
            }

            // apply jacobi scaling preconditioning. All the downstream operations
            // (even the diagonal/elliptical scaling) see the thusly scaled
            // jacobian. Only at the very end to unscale the parameters, we
            // reverse the scaling. To reverse that scaling, we multiply
            // with the same factor without inversion!
            // to the jacobian
            if let Some(jacobi_scaling) = jacobi_scaling.as_ref() {
                jacobian = try_opt!(jacobian.mul_diag_right(jacobi_scaling, Invert::No),
                            on_none = TerminationFailure::WrongDimensions(
                               "jacobi scaling and jacobian are incompatible (jacobian likely changed dimensions between iterations)"
                           ),
                            problem = problem
                    );
            }
            
            let jacobian_col_norms = jacobian.column_enorms();


            if self.use_elliptical_parameter_scaling {
                // see the MINPACK User guide, chapter 2.5 on scaling. In the
                // text they mention that they arbitrarily replace a zero
                // weighting by 1.
                diagonal_weights = Some(
                    jacobian_col_norms
                        .clone_owned()
                        // .replace_if_leq(T::ZERO, T::ONE),
                        .clamp(self.min_diagonal, self.max_diagonal)
                );
            }

            // some special sauce (see the iter == 1 / iter .EQ. 1 blocks in MINPACK)
            if is_first_iteration {
                // the norm of the (possibly scaled) parameters
                let param_norm = {
                    if let Some(diag) = diagonal_weights.as_ref() {
                        try_opt!(
                            params.clone().diag_mul_left(diag, Invert::No),
                            on_none = TerminationFailure::WrongDimensions(
                                "parameters have incompatible dimensions for weights"
                            ),
                            problem = problem
                        )
                        .enorm()
                    } else {
                        params.enorm()
                    }
                };
                delta = param_norm * self.factor;
                if delta.is_zero() {
                    delta = self.factor;
                }
                // DEBUG(georgios)
                delta = FromPrimitive::from_u16(10000).unwrap();
                debug_assert!(!delta.is_zero());
            }

            // compute new scaling matrix (if scaling is requested) and perform the scaling
            // note: if scaling is requested, the diagonal weights will be Some(...)
            if let Some(diag) = diagonal_weights.take() {
                let diag = try_opt!(
                    diag.elementwise_max(&jacobian_col_norms),
                    on_none = TerminationFailure::WrongDimensions(
                        "jacobian changed shape between iterations"
                    ),
                    problem = problem
                );

                // scaled jacobian is J' = J D^-1
                let scaled_jac = try_opt!(
                    jacobian.mul_diag_right(&diag, Invert::Yes),
                    on_none = TerminationFailure::WrongDimensions(
                        "jacobian and weights have incompatible dimensions"
                    ),
                    problem = problem
                );
                jacobian = scaled_jac;

                //TODO WARN: is that true ??? FIX??? No I think it's true
                // we're doing the gradient calculation afterwards with the
                // scaled or unscaled jacobian, which should give the correct
                // results
                // // scaled gradient is g' = D^-1 g
                // let scaled_grad = try_opt!(
                //     gradient.diag_mul_left(&diag, Invert::Yes),
                //     on_none = TerminationFailure::WrongDimensions(
                //         "gradient and weights have incompatible dimensions"
                //     ),
                //     problem = problem
                // );
                // gradient = scaled_grad;
                diagonal_weights = Some(diag);
            }

            let gradient = try_opt!(
                jacobian.tr_mulv(&residuals),
                on_none = TerminationFailure::WrongDimensions("J^T r"),
                problem = problem
            );
            // gtol check
            let gmax = try_opt!(
                gtol_calc(&jacobian_col_norms, &gradient, rnorm),
                on_none =
                    TerminationFailure::WrongDimensions("zero dimension for residuals or jacobian"),
                problem = problem
            );
            // println!("gmax = {:?}", gmax);

            if gmax <= self.gtol {
                return Ok((
                    problem,
                    MinimizationReport {
                        termination: TerminationReason::Converged(report::StoppingCriterion::Gtol),
                        number_of_evaluations: nfunc_evals,
                        objective_function,
                    },
                ));
            }


            // initialize the step solver with the given (unscaled) residuals, and the
            // (possibly scaled) gradient and (possibly scaled) jacobian.
            // Note that the returned step is thus p' = Dp.
            // print!("gradient: {:?}",gradient);
            let mut step_solver = try2!(
                S::init(jacobian, residuals.clone(), gradient),
                problem = problem
            );

            // inner loop
            'inner: loop {
                // again, note that the step is p' = Dp, i.e. the possibly scaled step
                let (dogleg_step, solver) =
                    try2!(step_solver.update_step(delta), problem = problem);
                step_solver = solver;
                // Convergence checks
                // println!("*** Step ***");
                // println!("step: {:?}", dogleg_step);

                // this is (like in MINPACK) the possibly scaled norm of p
                let DoglegStep {
                    // this is scaled p' = Dp
                    // if we have no scaling, this is just p
                    p: step_scaled,
                    // this is the norm of scaled p
                    p_norm: p_scaled_norm,
                    // the predicted reduction is independent of the scaling
                    // see the chapter on scaling in Nocedal&Wright using the definitions
                    // for scaled g, scaled J, and scaled p it turns out that
                    // all the D and D^-1 expressions will cancel out such
                    // that the predicted reduction is independent of scaling
                    // @note(geo-ant) #2 that we still need to normalize the predicted
                    // retuction by the current function norm
                    predicted_reduction,
                } = dogleg_step;

                // TODO(geo-ant): re-enable this???
                // // adjust the initial step bound on first iteration
                // if is_first_iteration {
                //     // it's correct to use the scaled norm here
                //     // cf. the MINPACK implementation of lmder
                //     delta = delta.min(p_scaled_norm);
                // }

                // get the new step candidate depending on whether we use scaling
                // or not. If we use scaling, we have to convert the step to unscaled
                // space
                let step = if let Some(diag) = diagonal_weights.as_ref() {
                    try_opt!(
                        step_scaled.diag_mul_left(diag, Invert::Yes),
                        on_none = TerminationFailure::WrongDimensions(
                            "parameter and weights have incompatible dimensions"
                        ),
                        problem = problem
                    )
                } else {
                    step_scaled
                };

                // inverse the jacobi scaling here, if it was applied
                let step =  
                if let Some(jacobi_scaling) = jacobi_scaling.as_ref() {
                    try_opt!(step.diag_mul_left(jacobi_scaling, Invert::No),
                        on_none = TerminationFailure::WrongDimensions("jacobi scaling and parameters have incompatible dimensions"),
                        problem = problem)
                } else {
                    step
                };

                // candidate for the new parameters
                // x_new = x + p
                let new_params = try_opt!(
                    // !!!!!!!!!!!!!!! we got to keep track of the current parameters!!!!!
                    // the problem parameters could be intermediate values that we discarded!!
                    // !!!!!!!!!!!!
                    // !!!!!!!!!!!!!!!!
                    step.scaled_add(T::ONE, &params),
                    on_none = TerminationFailure::WrongDimensions(
                        "parameters and step have incompatible dimensions"
                    ),
                    problem = problem
                );

                {
                    let mut problem_guard = reset_guard::update_params(&mut problem, new_params);
                    nfunc_evals += 1;

                    let new_residuals = try_opt!(
                        problem_guard.residuals(),
                        on_none = TerminationFailure::ResidualEval,
                        problem = problem,
                        guard = problem_guard
                    );
                    let new_rnorm = new_residuals.enorm();

                    // this is the same as in the minpack implementation
                    let actual_reduction = 
                        // this is WRONG because I'm not using the relative predicted
                        // T::ONE - Float::powi(new_rnorm / rnorm, 2)
                        T::P5 * (Float::powi(rnorm, 2) - Float::powi(new_rnorm, 2));

                    // this is also the same as in MINPACK
                    let ratio = if predicted_reduction != T::ZERO {
                        // actual_reduction / predicted_reduction
                        actual_reduction / predicted_reduction
                    } else {
                        -T::max_value()
                    };
                    // println!("ratio: {:?}", ratio);
                    // println!("predred: {:?}", predicted_reduction);
                    // println!("actred: {:?}", actual_reduction);
                    // println!("delta: {:?}",delta);
                    // println!("params: {:?}", params);

                    let accept_update = ratio >= T::P0001;

                    // println!("objective fn: {:?}", objective_function);
                    if accept_update {
                        // println!("update accepted");
                        rnorm = new_rnorm;
                        objective_function = T::P5 * rnorm.powi(2);
                        residuals = new_residuals;
                        params = problem_guard.params();
                        // defusing this is important here because otherwise
                        // the guard will reset the parameters on drop, which
                        // is only the correct thing to do if the parameters
                        // are not accepted.
                        problem_guard.defuse();
                    } 

                    // step expansion or shrinking logic. This is how the
                    // CERES solver does it in its dogleg implementation.
                    if accept_update {
                        if ratio <= T::P25 {
                            delta = delta * T::P5;
                        }

                        if ratio >= T::P75 {
                            delta = Float::max(delta, T::THREE* p_scaled_norm);
                        }
                    } else {
                        delta = delta*T::P5;
                    }

                    is_first_iteration = false;

                    // F-convergence check, see MINPACK user guide p. 22-24
                    if (FtolCheck {
                        predicted_reduction,
                        actual_reduction,
                        ratio,
                        tol: self.ftol,
                    })
                    .check()
                    {
                        drop(problem_guard);
                        return Ok((
                            problem,
                            MinimizationReport {
                                termination: TerminationReason::Converged(
                                    report::StoppingCriterion::Ftol,
                                ),
                                number_of_evaluations: nfunc_evals,
                                objective_function,
                            },
                        ));
                    }

                    // X-convergence check (for parameters), MINPACK user guide p. 23
                    let xnorm = match diagonal_weights.as_ref() {
                        Some(weights) => {
                            try_opt!(
                                params.diag_mul_left_enorm(weights),
                                on_none = TerminationFailure::WrongDimensions(
                                    "parameter and weights have incompatible dimensions"
                                ),
                                problem = problem,
                                guard = problem_guard
                            )
                        }
                        // params.diag_mul_left_enorm(weights).unwrap(),
                        None => params.enorm(),
                    };
                    let xtol_check = delta <= self.xtol * xnorm;

                    if xtol_check {
                        drop(problem_guard);
                        return Ok((
                            problem,
                            MinimizationReport {
                                termination: TerminationReason::Converged(
                                    report::StoppingCriterion::Xtol,
                                ),
                                number_of_evaluations: nfunc_evals,
                                objective_function,
                            },
                        ));
                    }

                    if nfunc_evals >= max_func_evals {
                        drop(problem_guard);
                        return Err(Error {
                            problem,
                            failure: TerminationFailure::LostPatience,
                        });
                    }

                    // repeat the tests for the convergence criteria with the machine
                    // epsilon. If those conditions are hit, it means no improvements
                    // are possible and the tolerances must be made bigger.
                    if gmax <= T::EPSMCH {
                        drop(problem_guard);
                        return Err(Error {
                            problem,
                            failure: TerminationFailure::NoImprovementPossible(
                                report::StoppingCriterion::Gtol,
                            ),
                        });
                    }

                    if (FtolCheck {
                        predicted_reduction,
                        actual_reduction,
                        ratio,
                        tol: T::EPSMCH,
                    })
                    .check()
                    {
                        drop(problem_guard);
                        return Err(Error {
                            problem,
                            failure: TerminationFailure::NoImprovementPossible(
                                report::StoppingCriterion::Ftol,
                            ),
                        });
                    }

                    // this is for xtol
                    if delta <= T::EPSMCH {
                        drop(problem_guard);
                        return Err(Error {
                            problem,
                            failure: TerminationFailure::NoImprovementPossible(
                                report::StoppingCriterion::Xtol,
                            ),
                        });
                    }
                    if accept_update {
                        break 'inner; // inner loop
                    }
                } // scope for setting new step and checking return conditions
            } //inner loop
        } //outer loop
    }
}

#[derive(Debug)]
/// helper structure to factor out the F-convergence check,
/// see MINPACK user guide p. 22-24
struct FtolCheck<T> {
    pub predicted_reduction: T,
    pub actual_reduction: T,
    pub ratio: T,
    pub tol: T,
}

impl<T: Float + MagicConst> FtolCheck<T> {
    fn check(self) -> bool {
        debug_assert_eq2!(self.ratio, self.actual_reduction / self.predicted_reduction);
        self.predicted_reduction <= self.tol
            && Float::abs(self.actual_reduction) <= self.tol
            && self.ratio <= MagicConst::TWO
    }
}
