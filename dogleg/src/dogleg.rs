use crate::dogleg::common::DoglegStepSolver;
use crate::dogleg::svd_impl::SvdDoglegSolver;
use dogleg_matx::{Colx, Matx, Scalex, Svdx, ToSvdx, TrMatVecMulx, TransformedVecNorm};
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, Dim, DimMin, DimSub, RealField, Scalar};
use num_traits::{ConstOne, Float};

mod common;
mod qr_impl;
mod svd_impl;

pub struct Dogleg<F> {
    /// initial radius of the trust region boundary
    delta_initial: F,
}

pub struct MinimizationReport;

// impl<T> Dogleg<T>
// where
//     T: RealField + Scalar + Copy + Float + ConstOne,
// {
//     pub fn minimize<M, N, P>(&self, problem: P) -> (P, MinimizationReport)
//     where
//         P: LevMarLeastSquaresProblem<T, M, N>,
//         <M as DimMin<N>>::Output: DimSub<Const<1>>,
//         M: Dim + DimMin<N>,
//         N: Dim,
//         DefaultAllocator: Allocator<M, N>,
//         DefaultAllocator: Allocator<N, M>,
//         DefaultAllocator: Allocator<M>,
//         DefaultAllocator: Allocator<N>,
//         DefaultAllocator: Allocator<M, <M as DimMin<N>>::Output>,
//         DefaultAllocator: Allocator<<M as DimMin<N>>::Output>,
//         DefaultAllocator: Allocator<<M as DimMin<N>>::Output, N>,
//         DefaultAllocator: Allocator<<<M as DimMin<N>>::Output as DimSub<Const<1>>>::Output>,
//     {
//         let jac = problem.jacobian().unwrap();
//         let res = problem.residuals().unwrap();
//         // @todo(geo) super stupid, but just so I can use it
//         // @todo(geo) REMOVE HACK FIX
//         let grad = jac.clone_owned().transpose() * &res;

//         let solver = SvdDoglegSolver::init(jac, res, grad).unwrap();
//         let (_step, _solver_new) = solver.calc_step(T::one()).unwrap();

//         // nonsense code, just to see if my abstractions work with the levmar
//         // stuff.
//         minimize_impl(
//             problem.jacobian().unwrap(),
//             problem.residuals().unwrap(),
//             self.delta_initial,
//         );
//         todo!()
//     }
// }

// fn minimize_with_solver<T,MMN,VM,VN,DS>(

//@todo(geo) change, this is just to see if my abstractions work
fn minimize_impl<T, Jac, Res>(
    jacobian: Jac,
    residuals: Res,
    delta_initial: T,
) -> Option<MinimizationReport>
where
    T: Scalar + RealField + Float + ConstOne,
    Jac: Matx<T>,
    Jac::Owned: TrMatVecMulx<T, Res, Output: Scalex<T>>,
    Jac::Owned: TransformedVecNorm<T, <Jac::Owned as TrMatVecMulx<T, Res>>::Output>,
    Jac::Owned: Clone + ToSvdx<T>,
    <Jac::Owned as ToSvdx<T>>::Svd: Svdx<T, Res>,
    Res: Colx<T> + Scalex<T>,
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
