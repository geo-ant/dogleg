use dogleg_matx::{Addx, Colx, Dotx, Matx, Scalex, Svdx, ToSvdx, TransformedVecNorm};
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dim, DimMin, DimSub, OMatrix, RealField, Scalar,
    Storage, SVD,
};
use num_traits::{float::TotalOrder, ConstOne, Float};

use crate::{
    dogleg::common::{
        dogleg_step, gtol_calc, DoglegComponents, DoglegStepSolver, DoglegStepSolverOLD,
    },
    utility::{enorm, enorm_squared},
};

use super::common::DoglegStep;

pub enum SvdDoglegSolver<T, MMN, VM, VN> {
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

impl<T, MMN, VN, VM> DoglegStepSolver<T, MMN, VM, VN> for SvdDoglegSolver<T, MMN, VM, VN>
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
    fn init(jacobian: MMN, residuals: VM, gradient: VN) -> Result<Self, crate::Error> {
        Ok(Self::Init {
            jacobian,
            residuals,
            gradient,
        })
    }

    fn calc_step(self, delta: T) -> Result<(DoglegStep<T, VN>, Self), crate::Error> {
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

pub struct SvdDoglegSolverOld;
impl<T, R, C> DoglegStepSolverOLD<T, R, C> for SvdDoglegSolverOld
where
    C: Dim + DimSub<Const<1>>,
    T: Scalar + Float + RealField + TotalOrder,
    R: Dim + DimMin<C, Output = C>,
    DefaultAllocator: Allocator<R>,
    DefaultAllocator: Allocator<R, C>,
    DefaultAllocator: Allocator<C>,
    DefaultAllocator: Allocator<C, C>,
    DefaultAllocator: Allocator<<C as DimSub<Const<1>>>::Output>,
{
    type Cache = OMatrix<T, R, C>;

    fn dogleg_components<S1>(
        jacobian: nalgebra::OMatrix<T, R, C>,
        residuals: &nalgebra::Vector<T, R, S1>,
        delta: T,
    ) -> Result<super::common::DoglegComponents<T, C, Self::Cache>, crate::Error>
    where
        S1: Storage<T, R>,
    {
        let g = jacobian.tr_mul(residuals);

        let _gtol_check_value = gtol_calc(&jacobian, &residuals);

        // if gtol_check_value <= gtol {
        //     return Ok(DoglegComponents::GtolSatisfied(gtol_check_value));
        // }

        let g_norm2 = enorm_squared(&g);
        // calculate ||J g||^2
        let jg_norm2 = enorm_squared(&(&jacobian * &g));

        // this is what we need to calculate p_u = u * g
        let p_u = g * (g_norm2 / jg_norm2);
        let pu_norm = enorm(&p_u);

        // this is an optimization that can save us from having to calculate the SVD
        if pu_norm < delta {
            return Ok(DoglegComponents::FirstSegmentInside {
                p_u,
                pu_norm,
                cached: jacobian.clone_owned(),
            });
        }

        let _svd = SVD::new_unordered(jacobian, true, true);

        todo!()
    }
}
