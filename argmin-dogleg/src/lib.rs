use argmin::{
    core::{ArgminFloat, CostFunction, Executor, Gradient, Hessian},
    solver::trustregion::Dogleg,
};
use argmin_math::ArgminInv;
use dogleg_matx::{magic_const::MagicConst, Colx};
use nalgebra::{DefaultAllocator, Dim, OMatrix, OVector, RealField, Vector};
use num_traits::{Float, FloatConst};
use std::{marker::PhantomData, sync::Mutex};

pub struct ArgminWrapper<P, T, M, N> {
    // this is stupid, but the interfaces of argmin and levenberg-marquardt differ
    // in how parameters are set and how they work with mutability
    pub inner: Mutex<P>,
    phantom: PhantomData<(T, M, N)>,
}

impl<P, T, M, N> ArgminWrapper<P, T, M, N> {
    pub fn new(problem: P) -> Self {
        Self {
            inner: Mutex::new(problem),
            phantom: PhantomData,
        }
    }
}

impl<P, T, M, N> CostFunction for ArgminWrapper<P, T, M, N>
where
    M: Dim,
    N: Dim,
    T: Float + Copy + RealField + MagicConst,
    P: levenberg_marquardt::LeastSquaresProblem<T, M, N>,
    DefaultAllocator: nalgebra::allocator::Allocator<N>,
    DefaultAllocator: nalgebra::allocator::Allocator<M>,
{
    type Param = Vector<T, N, P::ParameterStorage>;
    type Output = T;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let mut guard = self.inner.lock().unwrap();
        guard.set_params(param);

        Ok(T::P5
            * guard
                .residuals()
                .ok_or(argmin::core::Error::msg("residuals failed"))?
                .enorm())
    }
}

impl<P, T, M, N> Gradient for ArgminWrapper<P, T, M, N>
where
    M: Dim,
    N: Dim,
    T: Float + Copy + RealField + MagicConst,
    P: levenberg_marquardt::LeastSquaresProblem<T, M, N>,
    DefaultAllocator: nalgebra::allocator::Allocator<N>,
    DefaultAllocator: nalgebra::allocator::Allocator<M>,
    DefaultAllocator: nalgebra::allocator::Allocator<N, M>,
{
    type Param = <Self as CostFunction>::Param;
    type Gradient = OVector<T, N>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let mut guard = self.inner.lock().unwrap();
        guard.set_params(param);
        let jac = guard
            .jacobian()
            .ok_or(argmin::core::Error::msg("jacobian failed"))?;
        let r = guard
            .residuals()
            .ok_or(argmin::core::Error::msg("residuals failed"))?;
        Ok(jac.transpose() * r)
    }
}

impl<P, T, M, N> Hessian for ArgminWrapper<P, T, M, N>
where
    M: Dim,
    N: Dim,
    T: Float + Copy + RealField + MagicConst,
    P: levenberg_marquardt::LeastSquaresProblem<T, M, N>,
    DefaultAllocator: nalgebra::allocator::Allocator<N>,
    DefaultAllocator: nalgebra::allocator::Allocator<M>,
    DefaultAllocator: nalgebra::allocator::Allocator<N, M>,
    DefaultAllocator: nalgebra::allocator::Allocator<N, N>,
{
    // type ResidualStorage: RawStorageMut<F, M> + Storage<F, M> + IsContiguous;
    // type JacobianStorage: RawStorageMut<F, M, N> + Storage<F, M, N> + IsContiguous;
    // type ParameterStorage: RawStorageMut<F, N> + Storage<F, N> + IsContiguous + Clone;
    type Param = <Self as CostFunction>::Param;
    type Hessian = OMatrix<T, N, N>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, argmin::core::Error> {
        let mut guard = self.inner.lock().unwrap();
        guard.set_params(param);
        let jac = guard
            .jacobian()
            .ok_or(argmin::core::Error::msg("jacobian failed"))?;
        Ok(jac.transpose() * jac)
    }
}

pub fn run_dogleg<P, T, M, N>(problem: P, initial: Vector<T, N, P::ParameterStorage>)
where
    M: Dim,
    N: Dim,
    T: Float + Copy + RealField + MagicConst + FloatConst + ArgminFloat,
    P: levenberg_marquardt::LeastSquaresProblem<T, M, N>,
    DefaultAllocator: nalgebra::allocator::Allocator<N>,
    DefaultAllocator: nalgebra::allocator::Allocator<M>,
    DefaultAllocator: nalgebra::allocator::Allocator<N, M>,
    DefaultAllocator: nalgebra::allocator::Allocator<N, N>,
    OMatrix<T, N, N>: ArgminInv<T>,
{
    let subproblem = argmin::solver::trustregion::Dogleg::new();
    let trustregion = argmin::solver::trustregion::TrustRegion::new(subproblem);
    let exec = Executor::new(ArgminWrapper::new(problem), trustregion)
        .configure(|state| state.param(initial).max_iters(100));
    exec.run()?
    todo!()
}
