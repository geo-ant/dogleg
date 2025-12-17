use super::common::DoglegStep;
use crate::{
    dogleg::{
        common::{dogleg_step, DoglegStepSolver},
        report::TerminationFailure,
    },
    MagicConst,
};
use dogleg_matx::{Addx, Colx, Dotx, Matx, Scalex, Svdx, ToSvdx, TransformedVecNorm};
use num_traits::{ConstZero, Float};

#[cfg(feature = "assert2")]
use assert2::debug_assert;

/// Dogleg solver using singular value decomposition internally, which is not
/// the cheapest way to calculate the step, but it's available on both `faer`
/// and lapack-free `nalgebra`.
#[derive(Debug, Clone, PartialEq)]
pub enum SvdStepSolver<T, MMN, VM, VN> {
    Init {
        // (scaled) Jacobian (matrix of size MxN)
        jacobian: MMN,
        // (not scaled) residuals (column vector with M elements)
        residuals: VM,
        // (scaled) gradient (column vector with N elements)
        gradient: VN,
    },
    Cached(SvdSolverCache<T, MMN, VN>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SvdSolverCache<T, MMN, VN> {
    /// Jacobian of the problem, an MxN matrix
    jacobian: MMN,
    /// we don't save pu explicitly, but `u` such that `u*g = pu`
    u: T,
    /// the gradient J^T r
    g: VN,
    /// ||g||
    g_norm: T,
    /// the pb part of the dogleg step
    pb: VN,
    /// ||pb||
    pb_norm: T,
    /// @todo remove!!
    rank: usize,
}

impl<T, MMN, VN, VM> DoglegStepSolver<T> for SvdStepSolver<T, MMN, VM, VN>
where
    T: Float + ConstZero + std::fmt::Debug + MagicConst,
    MMN: Matx<T> + TransformedVecNorm<T, VN>,
    MMN::Owned: ToSvdx<T>,
    // the output of the least squares solution of `j pb = -r` is a vector of dim N
    // just like the gradient
    <MMN::Owned as ToSvdx<T>>::Svd: Svdx<T, VM, Output = VN>,
    VN: Colx<T> + Scalex<T>,
    VM: Colx<T> + Scalex<T>,
    VN: Colx<T, Owned = VN> + Addx<T, VN> + Dotx<T, VN> + Scalex<T>,
    VN::Owned: Scalex<T> + Addx<T, VN> + Colx<T>,
{
    type Jacobian = MMN;
    type Gradient = VN;
    type Residuals = VM;

    fn init(jacobian: MMN, residuals: VM, gradient: VN) -> Result<Self, TerminationFailure> {
        Ok(Self::Init {
            jacobian,
            residuals,
            gradient,
        })
    }

    fn update_step(self, delta: T) -> Result<(DoglegStep<T, VN>, Self), TerminationFailure> {
        // if we haven't already cached the calculations, do them now
        let cached = match self {
            Self::Init {
                jacobian,
                gradient,
                residuals,
            } => {
                // ||J g||
                let jg_norm = jacobian
                    .mulv_enorm(&gradient)
                    .ok_or(TerminationFailure::WrongDimensions("gradient and jacobian"))?;
                // ||g||
                let g_norm = gradient.enorm();

                let jacobian_clone = jacobian.clone_owned();
                // we need to enforce this somewhere else! My assumptions might
                // not hold for underdetermined problems.
                debug_assert!(jacobian.nrows().unwrap() >= jacobian.ncols().unwrap());
                //@todo remove
                let matdim = jacobian_clone
                    .nrows()
                    .unwrap()
                    .min(jacobian_clone.ncols().unwrap());
                // @todo!! maybe this isn't actually true, since the number of
                // rows can be less than the number of cols
                // @todo also remove!!
                debug_assert!(gradient.dim().unwrap() == matdim);

                let svd = jacobian_clone
                    .calc_svd()
                    .ok_or(TerminationFailure::Numerical("svd"))?;
                let minus_r = residuals.scale(-T::ONE);
                let pb = svd
                    .solve_lsqr(&minus_r)
                    .ok_or(TerminationFailure::Numerical("lsqr solve"))?;
                let pb_norm = pb.enorm();
                let u = Float::powi(g_norm, 2) / Float::powi(jg_norm, 2);
                let rank = svd.rank();
                SvdSolverCache {
                    u,
                    g: gradient,
                    g_norm,
                    pb,
                    jacobian,
                    pb_norm,
                    rank,
                }
            }
            Self::Cached(cached) => cached,
        };

        // at this point we have to pick the correct combination of steps,
        // given pb and pu
        // @todo(geo-ant) this method can be made more efficient by also
        // providing the already calculated norms
        let pu = cached.g.clone_owned().scale(cached.u);
        let p = dogleg_step(&pu, &cached.pb, delta)?;

        // predicted reduction is
        // m(0) - m(p) = -g^T p - 1/2 ||J p||^2
        // (mathematically, this must always be a positive number, so that should
        // be a good sanity check)
        // But we don't save g here, but we know

        let predicted_reduction = -cached
            .g
            .dot(&p)
            .ok_or(TerminationFailure::WrongDimensions("gradient and step"))?
            - T::P5
                * Float::powi(
                    cached
                        .jacobian
                        .mulv_enorm(&p)
                        .ok_or(TerminationFailure::WrongDimensions("jacobian and step"))?,
                    2,
                );

        // @note(geo-ant) mathematically, the predicted reduction is always >= zero,
        // but due to numerical reasons, this can have very small values.
        debug_assert!(
            predicted_reduction >= -T::EPSMCH,
            "rank is {} of {}",
            cached.rank,
            cached.g.dim().unwrap()
        );
        let predicted_reduction = predicted_reduction.abs();

        let p_norm = p.enorm();
        let step = DoglegStep {
            p,
            p_norm,
            predicted_reduction,
        };

        Ok((step, Self::Cached(cached)))
    }
}
