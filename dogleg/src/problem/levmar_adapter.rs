use crate::levenberg_marquardt::LeastSquaresProblem as LevMarLeastSquaresProblem;
use crate::problem::LeastSquaresProblem;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, Dim, IsContiguous, Matrix, Owned, RawStorageMut,
    RealField, Storage, Vector,
};
use num_traits::Float;
use std::marker::PhantomData;

/// An adapter type so that any type that implements the
/// [`LeastSquaresProblem`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/trait.LeastSquaresProblem.html)
/// trait can be immediately used with the dogleg minimizer.
///
/// Simply create an a `LevMarAdapter` from it and throw it to the dogleg
/// minimizer and you're good to go.
#[derive(Debug)]
pub struct LevMarAdapter<P, T, M, N>
where
    T: Copy + nalgebra::ComplexField,
    P: LevMarLeastSquaresProblem<T, M, N>,
    N: nalgebra::Dim,
    M: nalgebra::Dim,
{
    pub inner: P,
    phantom: PhantomData<(T, M, N)>,
}

impl<P, T, M, N> Clone for LevMarAdapter<P, T, M, N>
where
    T: Copy + nalgebra::ComplexField,
    P: LevMarLeastSquaresProblem<T, M, N> + Clone,
    N: nalgebra::Dim,
    M: nalgebra::Dim,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            phantom: self.phantom.clone(),
        }
    }
}

impl<P, T, M, N> Copy for LevMarAdapter<P, T, M, N>
where
    T: Copy + nalgebra::ComplexField,
    P: LevMarLeastSquaresProblem<T, M, N> + Copy,
    N: nalgebra::Dim,
    M: nalgebra::Dim,
{
}

impl<P, T, M, N> LevMarAdapter<P, T, M, N>
where
    T: Copy + nalgebra::ComplexField,
    P: LevMarLeastSquaresProblem<T, M, N>,
    N: nalgebra::Dim,
    M: nalgebra::Dim,
{
    /// same as `from`
    pub fn new(problem: P) -> Self {
        Self::from(problem)
    }
}

/// wrap a problem that can be solved with the
/// [`LevenbergMarquardt`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/struct.LevenbergMarquardt.html)
/// minimizer from the `levenberg-marquardt` crate.
impl<P, T, M, N> From<P> for LevMarAdapter<P, T, M, N>
where
    T: Copy + nalgebra::ComplexField,
    P: LevMarLeastSquaresProblem<T, M, N>,
    N: nalgebra::Dim,
    M: nalgebra::Dim,
{
    fn from(problem: P) -> Self {
        Self {
            inner: problem,
            phantom: PhantomData,
        }
    }
}

impl<T, P, M, N> LeastSquaresProblem<T> for LevMarAdapter<P, T, M, N>
where
    T: Copy + RealField + Float,
    N: Dim,
    M: Dim,
    P: LevMarLeastSquaresProblem<T, M, N, ParameterStorage = Owned<T, N>>,
    P::ResidualStorage: RawStorageMut<T, M> + Storage<T, M> + IsContiguous + std::fmt::Debug,
    P::JacobianStorage: RawStorageMut<T, M, N> + Storage<T, M, N> + IsContiguous,
    DefaultAllocator: Allocator<N>,
    DefaultAllocator: Allocator<M>,
    DefaultAllocator: Allocator<M, N>,
{
    type Residuals = Vector<T, M, P::ResidualStorage>;
    type Parameters = Vector<T, N, P::ParameterStorage>;
    // we always return the owned type here due to implementation details in the
    // dogleg crate.
    type Jacobian = Matrix<T, M, N, Owned<T, M, N>>;

    fn set_params(&mut self, x: Self::Parameters) {
        self.inner.set_params(&x)
    }

    fn params(&self) -> Self::Parameters {
        self.inner.params()
    }

    fn residuals(&self) -> Option<Self::Residuals> {
        self.inner.residuals()
    }

    fn jacobian(&self) -> Option<Self::Jacobian> {
        self.inner.jacobian().map(|m| m.into_owned())
    }
}
