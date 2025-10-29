use crate::problem::LeastSquaresProblem;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, Dim, DimMin, DimSub, RealField, Scalar, SVD};
use num_traits::{ConstOne, Float};

mod common;
mod qr_impl;
mod svd_impl;

pub struct Dogleg<F> {
    /// initial radius of the trust region boundary
    delta_initial: F,
}

pub struct MinimizationReport;

impl<F> Dogleg<F>
where
    F: RealField + Scalar + Copy + Float + ConstOne,
{
    pub fn minimize<M, N, P>(&self, problem: P) -> (P, MinimizationReport)
    where
        P: LeastSquaresProblem<F, M, N>,
        M: Dim + DimMin<N>,
        N: Dim,
        DefaultAllocator: Allocator<M, N>,
        DefaultAllocator: Allocator<M>,
        DefaultAllocator: Allocator<N>,
        DefaultAllocator: Allocator<M, <M as DimMin<N>>::Output>,
        DefaultAllocator: Allocator<<M as DimMin<N>>::Output>,
        DefaultAllocator: Allocator<<M as DimMin<N>>::Output, N>,
        <M as DimMin<N>>::Output: DimSub<Const<1>>,
        DefaultAllocator: Allocator<<<M as DimMin<N>>::Output as DimSub<Const<1>>>::Output>,
    {
        minimize_impl(problem, self.delta_initial)
    }
}

//@todo(geo) change, this is just to see if my abstractions work
pub fn minimize_impl<M, N, P, F>(problem: P, delta_initial: F) -> (P, MinimizationReport)
where
    F: Scalar + RealField + Float + ConstOne,
    P: LeastSquaresProblem<F, M, N>,
    M: Dim + DimMin<N>,
    N: Dim,
    DefaultAllocator: Allocator<M, N>,
    DefaultAllocator: Allocator<M>,
    DefaultAllocator: Allocator<N>,
    DefaultAllocator: Allocator<M, <M as DimMin<N>>::Output>,
    DefaultAllocator: Allocator<<M as DimMin<N>>::Output>,
    DefaultAllocator: Allocator<<M as DimMin<N>>::Output, N>,
    <M as DimMin<N>>::Output: DimSub<Const<1>>,
    DefaultAllocator: Allocator<<<M as DimMin<N>>::Output as DimSub<Const<1>>>::Output>,
{
    // J: Jacobian matrix
    // we have to call into_owned here unfortunately, because all the solvers that we use
    // will actually need the matrix to be Owned.
    let j = problem.jacobian().unwrap().into_owned();
    // r: residual vector. Actually, this contains -r
    let r = -problem.residuals().unwrap();

    // this is the gradient g = J^T r.
    let g = j.tr_mul(&r);
    let alpha = g.norm_squared() / (&j * &g).norm_squared();

    let pu = g * alpha;
    let delta = delta_initial;

    let pu_norm_squared = pu.norm_squared();
    let d2 = Float::powi(delta, 2);

    let p_star;
    if pu_norm_squared >= d2 {
        let tau = delta / pu.norm();
        p_star = pu * tau;
    } else {
        let j_owned = j.clone_owned();

        //PERF: we can actually reuse this computation in the case where we don't accept the step.
        let svd = SVD::try_new_unordered(
            j_owned,
            true,
            true,
            // these are the default arguments that are passed to the downstream
            // nalgebra call for SVD::new_unordered(...)
            F::epsilon() * F::from_u8(5).unwrap(),
            0,
        )
        .unwrap();
        let pb = svd.solve(&r, F::epsilon()).unwrap();

        if pb.norm_squared() <= d2 {
            p_star = pb;
        } else {
            let a = pu_norm_squared;
            let b = pu.dot(&pb) - pu_norm_squared;
            let pb_minus_pu = pb - &pu;
            let c = (pb_minus_pu).norm_squared();

            // maybe check for problems here??
            // if c < Float::epsilon() {
            //     todo!()
            // }

            let d = d2;
            let b_c = b / c;
            let tau = F::ONE - b_c + Float::sqrt((d - a) / c + Float::powi(b_c, 2));
            p_star = pu + pb_minus_pu * tau;
        }
    }

    let _x_next = problem.params() + p_star;
    todo!()
}
