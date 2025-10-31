use crate::levenberg_marquardt::LeastSquaresProblem as LevMarLeastSquaresProblem;
use crate::problem::LeastSquaresProblem;
use nalgebra::{
    allocator::Allocator, ComplexField, DefaultAllocator, Dim, IsContiguous, Matrix, RawStorageMut,
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
pub struct LevMarAdapter<P, T, M, N>
where
    T: Copy + nalgebra::ComplexField,
    P: LevMarLeastSquaresProblem<T, M, N>,
    N: nalgebra::Dim,
    M: nalgebra::Dim,
{
    problem: P,
    phantom: PhantomData<(T, M, N)>,
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
            problem,
            phantom: PhantomData::default(),
        }
    }
}

impl<T, P, M, N> LeastSquaresProblem<T> for LevMarAdapter<P, T, M, N>
where
    T: ComplexField + Copy + RealField + Float,
    N: Dim,
    M: Dim,
    P: LevMarLeastSquaresProblem<T, M, N>,
    P::ResidualStorage: RawStorageMut<T, M> + Storage<T, M> + IsContiguous,
    P::JacobianStorage: RawStorageMut<T, M, N> + Storage<T, M, N> + IsContiguous,
    P::ParameterStorage: RawStorageMut<T, N> + Storage<T, N> + IsContiguous + Clone,
    DefaultAllocator: Allocator<N>,
    DefaultAllocator: Allocator<M>,
    DefaultAllocator: Allocator<M, N>,
{
    type Residuals = Vector<T, M, P::ResidualStorage>;
    type Parameters = Vector<T, N, P::ParameterStorage>;
    type Jacobian = Matrix<T, M, N, P::JacobianStorage>;

    fn set_params(&mut self, x: Self::Parameters) {
        self.problem.set_params(&x)
    }

    fn params(&self) -> Self::Parameters {
        self.problem.params()
    }

    fn residuals(&self) -> Option<Self::Residuals> {
        self.problem.residuals()
    }

    fn jacobian(&self) -> Option<Self::Jacobian> {
        self.problem.jacobian()
    }
}
