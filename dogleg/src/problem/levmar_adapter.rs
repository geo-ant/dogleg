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
/// trait from the [`levenberg-marquardt`](https://crates.io/crates/levenberg-marquardt)
/// crate can be immediately used with the dogleg minimizer.
///
/// Simply create a `LevMarAdapter` from the problem and use that in the
/// [`Dogleg`](crate::Dogleg) minimizer function.
///
/// **NOTE Performance**: The adapter returns an owned Jacobian matrix, which
/// which _is not_ a problem if the original problem also returns an owned
/// Jacobian, which is a very typical use case. Otherwise, this will incure
/// a performance hit due to the memory being copied.
///
/// ## Example
///
/// ```rust
/// # use nalgebra::U1;
/// # use nalgebra::Owned;
/// # fn foo(
/// #     lm_problem: impl levenberg_marquardt::LeastSquaresProblem<
/// #         f64,
/// #         U1,
/// #         U1,
/// #         ResidualStorage = nalgebra::Owned<f64, U1, U1>,
/// #         ParameterStorage = Owned<f64, U1>,
/// #         ResidualStorage = Owned<f64, U1>,
/// #     >,
/// # ) {
/// # use dogleg::Dogleg;
/// # use dogleg::LevMarAdapter;
/// // lm_problem implements levenberg_marquardt::LeastSquaresProblem
///
/// let result = Dogleg::new().minimize(LevMarAdapter::from(lm_problem));
/// # }
/// ```
#[derive(Debug)]
pub struct LevMarAdapter<P, T, M, N> {
    /// the wrapped `levenberg-marquardt` problem.
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
    // NOTE PERF(geo-ant) I want to change that, but I'll due that when I refactor
    // the ownership in the minimization logic.
    type Jacobian = Matrix<T, M, N, Owned<T, M, N>>;

    #[inline(always)]
    fn set_params(&mut self, x: Self::Parameters) {
        self.inner.set_params(&x)
    }

    #[inline(always)]
    fn params(&self) -> Self::Parameters {
        self.inner.params()
    }

    #[inline(always)]
    fn residuals(&self) -> Option<Self::Residuals> {
        self.inner.residuals()
    }

    #[inline(always)]
    fn jacobian(&self) -> Option<Self::Jacobian> {
        self.inner.jacobian().map(|m| m.into_owned())
    }
}
