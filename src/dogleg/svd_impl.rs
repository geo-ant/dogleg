use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dim, DimMin, DimSub, OMatrix, RealField, Scalar,
    Storage, SVD,
};
use num_traits::{float::TotalOrder, Float};

use crate::{
    dogleg::common::{gtol_calc, DoglegComponents, DoglegComponentsSolver},
    utility::{enorm, enorm_squared},
};
pub struct SvdDoglegSolver;

impl<T, R, C> DoglegComponentsSolver<T, R, C> for SvdDoglegSolver
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
        gtol: T,
    ) -> Result<super::common::DoglegComponents<T, C, Self::Cache>, crate::Error>
    where
        S1: Storage<T, R>,
    {
        let g = jacobian.tr_mul(residuals);

        let gtol_check_value = gtol_calc(&jacobian, &residuals);

        if gtol_check_value <= gtol {
            return Ok(DoglegComponents::GtolSatisfied(gtol_check_value));
        }

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
                cached: todo!(),
            });
        }

        let svd = SVD::new_unordered(jacobian, true, true);

        todo!()
    }
}
