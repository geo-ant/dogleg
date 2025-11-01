use crate::Error;
use crate::LeastSquaresProblem;
use crate::MagicConst;
use crate::TerminationFailure;
use dogleg_matx::{Colx, Matx, Scalex, Svdx, ToSvdx, TrMatVecMulx, TransformedVecNorm};
use num_traits::Float;
use std::num::NonZero;

mod common;
mod hack;
mod qr_impl;
mod svd_impl;

pub mod report;
pub use common::DoglegStep;
pub use common::DoglegStepSolver;
pub use report::MinimizationReport;
pub use report::TerminationReason;

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
                    termination : $failure
                });
            }
        }
    };

    // for calls like jacobian.mul_tr(&residuals)
    ($ident:ident . $function:ident ($($tokens:tt)*), on_none
         = $failure:expr, problem = $problem:ident) => {
        match $ident. $function ($($tokens)*) {
            Some(val) => val,
            None => {
                return Err($crate::Error {
                    problem: $problem,
                    termination : $failure
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
    /// Nocedal & Wright, p 95-97
    scale_diag: bool,
    /// Used to calculate the maximum number of function evals (a stopping
    /// criterion) based on the problem
    patience: u64,
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
            factor: T::ONE_E2,
            scale_diag: true,
            patience: 100,
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
    /// Whether to apply diagonal rescaling of variables, see e.g.
    /// Nocedal & Wright p 95-97
    pub fn with_scale_diag(self, scale_diag: bool) -> Self {
        Self { scale_diag, ..self }
    }
}

/// type of J^T r
type GradType<T, J, R> = <J as TrMatVecMulx<T, R>>::Output;

// use nalgebra::constraint::AreMultipliable;
// use nalgebra::constraint::ShapeConstraint;
// use nalgebra::iter;
// use nalgebra::ClosedAddAssign;
// use nalgebra::Const;
// use nalgebra::Dim;
// use nalgebra::DimMin;
// use nalgebra::DimSub;
// use nalgebra::RealField;
// use nalgebra::Scalar;
// use nalgebra::U1;
impl<T> Dogleg<T> {
    //@todo(geo) this works with only nalgebra trait bounds
    // pub fn min_levmar<P, M, N>(self, p: P)
    // where
    //     P: levenberg_marquardt::LeastSquaresProblem<T, M, N>,
    //     T: nalgebra::Scalar + Copy + RealField + Float + MagicConst,
    //     T: Scalar + RealField + Float + ClosedAddAssign + Copy + ConstOne,
    //     M: Dim,
    //     N: Dim,
    //     M: DimMin<N, Output = N>,
    //     nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<M>,
    //     nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<M, N>,
    //     nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<M>,
    //     nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<N>,
    //     ShapeConstraint: AreMultipliable<N, M, M, U1>,
    //     nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<M, <M as DimMin<N>>::Output>,
    //     nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<<M as DimMin<N>>::Output>,
    //     nalgebra::DefaultAllocator:
    //         nalgebra::allocator::Allocator<<M as nalgebra::DimMin<N>>::Output, N>,
    //     <M as DimMin<N>>::Output: DimSub<U1>,
    //     nalgebra::DefaultAllocator:
    //         nalgebra::allocator::Allocator<<<M as DimMin<N>>::Output as DimSub<U1>>::Output>,
    //     N: DimSub<Const<1>>,
    // {
    //     let adapter = LevMarAdapter::new(p);
    //     self.minimize_generic::<SvdStepSolver<_, _, _, _>, _>(adapter);
    // }

    pub fn minimize_generic<S, P>(&self, problem: P) -> Result<(P, MinimizationReport<T>), Error<P>>
    where
        T: Float + MagicConst,
        P: LeastSquaresProblem<T>,
        // see below, we require the gradient, i.e. the of J^T r to be the owned type of P.
        // This should barely be a restriction...
        S: DoglegStepSolver<
            T,
            Jacobian = P::Jacobian,
            Gradient = GradType<T, P::Jacobian, P::Residuals>,
            Residuals = P::Residuals,
        >,
        // for max func eval calculation
        u64: TryFrom<<P::Parameters as Colx<T>>::Dim, Error: std::fmt::Debug>,
        // for calculating the gradient g = J^T r
        // @note(geo-ant) what this means is: (J^T * r).to_owned() has the
        // same type as P::Parameters::Owned, which should not be a restriction in practice
        // because the dimensions of J^T r and the parameters must be identical
        P::Jacobian: TrMatVecMulx<T, P::Residuals>,
        GradType<T, P::Jacobian, P::Residuals>: Colx<T, Owned = <P::Parameters as Colx<T>>::Owned>,
    {
        // @todo(geo-ant) maybe refactor this at a later date, but for the
        // intended use of this library, the number of parameters being near
        // the u64 limit is completely unreasonable, hence panicking here is
        // completely fine. If you have that many parameters, you probably
        // shouldn't be using this algorithm anyway...
        // @note(geo-ant) this whole overflow checking is very pedantic,
        // but what the heck...
        let (n_plus_one, overflow) = u64::try_from(problem.params().dim())
            .expect("too many parameters for dogleg solver")
            .overflowing_add(1);
        let (max_func_evals, overflow2) = self.patience.overflowing_mul(n_plus_one);
        if overflow || overflow2 {
            panic!("too many parameters for dogleg solver");
        }

        let mut nfunc_evals: u64 = 0;
        let mut iter = 0;

        // @todo(geo-ant)
        // the trust region radius. In the first iteration of the algorithm,
        // this will be set to something useful, so initializing with factor
        // zero here is fine. But we can only calculate this after having
        // calculated the jacobian (and thus the scaling matrix) and pulling
        // this outside makes the loop diverge from the MINPACK implementation,
        // which I'm sticking to for now.
        let mut delta = T::zero();

        // outer loop
        loop {
            if nfunc_evals >= max_func_evals {
                return Err(Error {
                    problem,
                    termination: TerminationFailure::LostPatience,
                });
            }

            let residuals = try_opt!(
                problem.residuals(),
                on_none = TerminationFailure::ResidualEval
            );

            let rnorm = residuals.enorm();

            if rnorm.is_zero() {
                return Ok((
                    problem,
                    MinimizationReport {
                        termination: TerminationReason::ResidualsZero,
                        number_of_evaluations: nfunc_evals,
                        objective_function: T::P5 * rnorm,
                    },
                ));
            }

            nfunc_evals += 1;
            let jacobian = try_opt!(
                problem.jacobian(),
                on_none = TerminationFailure::JacobianEval
            );

            if iter == 0 {
                //@todo(geo) add scaling for parameters
                let scaled_params = problem.params();
                delta = scaled_params.enorm() * self.factor;
                if delta.is_zero() {
                    delta = self.factor;
                }
            }

            let gradient = try_opt!(
                jacobian.tr_mulv(&residuals),
                on_none = TerminationFailure::WrongDimensions("J^T r"),
                problem = problem
            );

            let mut step_solver = S::init(jacobian, residuals, gradient).unwrap();

            let (step, solver) = step_solver.calc_step(delta).unwrap();
            step_solver = solver;

            if iter > 10 {
                break;
            }

            let params = problem.params().into_owned();

            //@note(geo-ant)
            // this check is also completely overblown, but can possibly help me spot bugs
            assert!(iter < u64::MAX - 1, "iteration limit reached");
            iter += 1;
        }

        todo!()
    }
}

//@todo(geo) change, this is just to see if my abstractions work
fn minimize_impl<T, MMN, VM>(
    jacobian: MMN,
    residuals: VM,
    delta_initial: T,
) -> Option<MinimizationReport<T>>
where
    T: Float + MagicConst,
    MMN: Matx<T>,
    MMN::Owned: TrMatVecMulx<T, VM, Output: Scalex<T>>,
    MMN::Owned: TransformedVecNorm<T, <MMN::Owned as TrMatVecMulx<T, VM>>::Output>,
    MMN::Owned: Clone + ToSvdx<T>,
    <MMN::Owned as ToSvdx<T>>::Svd: Svdx<T, VM>,
    VM: Colx<T> + Scalex<T>,
{
    // J: Jacobian matrix
    // we have to call into_owned here unfortunately, because all the solvers that we use
    // will actually need the matrix to be Owned.
    let j = jacobian.into_owned();
    // r: residual vector. Actually, this contains -r
    let r = residuals.scale(-T::ONE);

    // this is the gradient g = J^T r.
    let g = j.tr_mulv(&r)?;
    let alpha = Float::powi(g.enorm(), 2) / Float::powi(j.mulv_enorm(&g)?, 2);

    // (&j * &g).norm_squared();

    let pu = g.scale(alpha);
    let delta = delta_initial;

    let pu_norm = pu.enorm();
    let _d2 = Float::powi(delta, 2);

    let _p_star;
    if pu_norm >= delta {
        let tau = delta / pu_norm;
        _p_star = pu.scale(tau);
    } else {
        let j_owned = j.clone();

        let svd = j_owned.calc_svd().unwrap();

        //PERF: we can actually reuse this computation in the case where we don't accept the step.
        // let svd = SVD::try_new_unordered(
        //     j_owned,
        //     true,
        //     true,
        //     // these are the default arguments that are passed to the downstream
        //     // nalgebra call for SVD::new_unordered(...)
        //     F::epsilon() * F::from_u8(5).unwrap(),
        //     0,
        // )
        // .unwrap();
        let _pb = svd.solve_lsqr(&r).unwrap();
    }
    todo!()
}
