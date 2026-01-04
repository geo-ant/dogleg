use argmin::{
    argmin_error,
    core::{ArgminFloat, CostFunction, Executor, Gradient, Hessian, State},
    solver::trustregion::{Dogleg, TrustRegion},
};
use argmin_math::ArgminInv;
use dogleg_matx::{magic_const::MagicConst, Colx};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, RealField, Vector};
use num_traits::{Float, FloatConst};
use std::{marker::PhantomData, sync::Mutex};

/// Allows us to use a `levenberg-marquardt::LeastSquaresProblem` in argmin.
/// This implementation IS NOT OPTIMIZED FOR EFFICIENCY. DO NOT USE FOR BENCHMARKS.
pub struct ArgminLevMarAdapter<P, T, M, N> {
    // this is stupid, but the interfaces of argmin and levenberg-marquardt differ
    // in how parameters are set and how they work with mutability
    pub inner: Mutex<P>,
    phantom: PhantomData<(T, M, N)>,
}

impl<P, T, M, N> ArgminLevMarAdapter<P, T, M, N> {
    pub fn new(problem: P) -> Self {
        Self {
            inner: Mutex::new(problem),
            phantom: PhantomData,
        }
    }
}

impl<P, T, M, N> CostFunction for ArgminLevMarAdapter<P, T, M, N>
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
            * Float::powi(
                guard
                    .residuals()
                    .ok_or(argmin::core::Error::msg("residuals failed"))?
                    .enorm(),
                2,
            ))
    }
}

impl<P, T, M, N> Gradient for ArgminLevMarAdapter<P, T, M, N>
where
    M: Dim,
    N: Dim,
    T: Float + Copy + RealField + MagicConst,
    P: levenberg_marquardt::LeastSquaresProblem<T, M, N, ParameterStorage = nalgebra::Owned<T, N>>,
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

impl<P, T, M, N> Hessian for ArgminLevMarAdapter<P, T, M, N>
where
    M: Dim,
    N: Dim,
    T: Float + Copy + RealField + MagicConst,
    P: levenberg_marquardt::LeastSquaresProblem<T, M, N, ParameterStorage = nalgebra::Owned<T, N>>,
    DefaultAllocator: nalgebra::allocator::Allocator<N>,
    DefaultAllocator: nalgebra::allocator::Allocator<M>,
    DefaultAllocator: nalgebra::allocator::Allocator<N, M>,
    DefaultAllocator: nalgebra::allocator::Allocator<N, N>,
{
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

pub struct ArgminReport<T> {
    pub objective_function: T,
}

/// run the dogleg minimizer in argmin on a `levenberg-marquardt::LeastSquaresProblem`
/// instance.
pub fn run_argmin_dogleg<P, T, M, N>(
    problem: P,
    initial: OVector<T, N>,
) -> Result<(P, ArgminReport<T>), argmin::core::Error>
where
    M: Dim,
    N: Dim,
    T: Float + Copy + RealField + MagicConst + FloatConst + ArgminFloat,
    P: levenberg_marquardt::LeastSquaresProblem<T, M, N, ParameterStorage = nalgebra::Owned<T, N>>,
    DefaultAllocator: nalgebra::allocator::Allocator<N>,
    DefaultAllocator: nalgebra::allocator::Allocator<M>,
    DefaultAllocator: nalgebra::allocator::Allocator<N, M>,
    DefaultAllocator: nalgebra::allocator::Allocator<N, N>,
    OMatrix<T, N, N>: ArgminInv<OMatrix<T, N, N>>,
{
    let subproblem = Dogleg::<T>::new();
    let trustregion = TrustRegion::new(subproblem);
    let exec = Executor::new(ArgminLevMarAdapter::new(problem), trustregion)
        .configure(|state| state.param(initial.clone()).max_iters(1000));
    let res = exec.run()?;

    let params = res
        .state()
        .get_best_param()
        .cloned()
        .ok_or(argmin::core::Error::msg("no best params"))?;

    let objective_function = res.state().get_best_cost();

    // extract the problem back out in an ugly, but safe fashion
    let mut prob = res.problem.problem.unwrap().inner.into_inner().unwrap();
    // just to make sure the optimal params are set
    prob.set_params(&params);
    Ok((prob, ArgminReport { objective_function }))
}
