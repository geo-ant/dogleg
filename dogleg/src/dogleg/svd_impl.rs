use super::common::DoglegStep;
use crate::dogleg::{
    common::{dogleg_step, DoglegStepSolver},
    report::TerminationFailure,
};
use dogleg_matx::{Addx, Colx, Dotx, Matx, Scalex, Svdx, ToSvdx, TransformedVecNorm};
use num_traits::{ConstOne, Float};

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
}

impl<T, MMN, VN, VM> DoglegStepSolver<T> for SvdStepSolver<T, MMN, VM, VN>
where
    T: ConstOne + Float,
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

    fn calc_step(self, delta: T) -> Result<(DoglegStep<T, VN>, Self), TerminationFailure> {
        // if we haven't already cached the calculations, do them now
        let cached = match self {
            Self::Init {
                jacobian,
                gradient,
                residuals,
            } => {
                // ||J g||
                let jg_norm = jacobian.mulv_enorm(&gradient).unwrap();
                // ||g||
                let g_norm = gradient.enorm();

                let jacobian_clone = jacobian.clone_owned();
                let svd = jacobian_clone.calc_svd().unwrap();
                let minus_r = residuals.scale(-T::ONE);
                let pb = svd.solve_lsqr(&minus_r).unwrap();
                let pb_norm = pb.enorm();
                let u = Float::powi(g_norm, 2) / Float::powi(jg_norm, 2);
                SvdSolverCache {
                    u,
                    g: gradient,
                    g_norm,
                    pb,
                    jacobian,
                    pb_norm,
                }
            }
            Self::Cached(cached) => cached,
        };

        // at this point we have to pick the correct combination of steps,
        // given pb and pu
        // @todo(geo-ant) this method can be made more efficient by also
        // providing the already calculated norms
        let pu = cached.g.clone_owned().scale(cached.u);
        let p = dogleg_step(&pu, &cached.pb, delta).unwrap();

        let half = T::ONE / (T::ONE + T::ONE); // 1/2
                                               // predicted reduction is
                                               // m(0) - m(p) = -g^T p - 1/2 ||J p||^2
                                               // (mathematically, this must always be a positive number, so that should
                                               // be a good sanity check)
                                               // But we don't save g here, but we know

        let predicted_reduction = -cached.g.dot(&p).unwrap()
            - half * Float::powi(cached.jacobian.mulv_enorm(&p).unwrap(), 2);
        debug_assert!(predicted_reduction.is_sign_positive() || predicted_reduction.is_zero());

        let p_norm = p.enorm();
        let step = DoglegStep {
            p,
            p_norm,
            predicted_reduction,
        };

        Ok((step, Self::Cached(cached)))
    }
}
