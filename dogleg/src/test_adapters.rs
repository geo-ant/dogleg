use crate::{levenberg_marquardt::LeastSquaresProblem as LevMarLeastSquaresProblem, LevMarAdapter};
use nalgebra::{allocator::Allocator, OVector, Owned, RawStorage, Scalar, U1};
use num_traits::{Float, Zero};
use std::marker::PhantomData;

pub trait AbstractLevMarProblemWrapper<P> {
    fn inner(self) -> P;
}

impl<P, T, M, N> AbstractLevMarProblemWrapper<P> for TestOnlyFaerLevMarAdapter<P, T, M, N> {
    fn inner(self) -> P {
        self.inner
    }
}

impl<P, T, M, N> AbstractLevMarProblemWrapper<P> for LevMarAdapter<P, T, M, N> {
    fn inner(self) -> P {
        self.inner
    }
}

#[derive(Debug)]
// WARN PERF not for benchmarking
/// USE FOR TESTING ONLY NOT FOR BENCHMARKING
/// This is like the levmar adapter, but for takes a levmar problem
/// and converts it to use faer matrices. Super dumb and slow!!
pub struct TestOnlyFaerLevMarAdapter<P, T, M, N> {
    /// the wrapped `levenberg-marquardt` problem.
    pub inner: P,
    phantom: PhantomData<(T, M, N)>,
}

impl<P, T, M, N> Clone for TestOnlyFaerLevMarAdapter<P, T, M, N>
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

impl<P, T, M, N> Copy for TestOnlyFaerLevMarAdapter<P, T, M, N>
where
    T: Copy + nalgebra::ComplexField,
    P: LevMarLeastSquaresProblem<T, M, N> + Copy,
    N: nalgebra::Dim,
    M: nalgebra::Dim,
{
}

/// wrap a problem that can be solved with the
/// [`LevenbergMarquardt`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/struct.LevenbergMarquardt.html)
/// minimizer from the `levenberg-marquardt` crate.
impl<P, T, M, N> From<P> for TestOnlyFaerLevMarAdapter<P, T, M, N>
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

impl<P, T, M, N> TestOnlyFaerLevMarAdapter<P, T, M, N>
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

impl<T, P, M, N> crate::LeastSquaresProblem<T> for TestOnlyFaerLevMarAdapter<P, T, M, N>
where
    T: Copy + nalgebra::ComplexField + Float + faer::traits::RealField,
    P: LevMarLeastSquaresProblem<T, M, N, ParameterStorage = Owned<T, N>>,
    N: nalgebra::Dim,
    M: nalgebra::Dim,
    nalgebra::DefaultAllocator: Allocator<N>,
{
    type Residuals = faer::Col<T>;
    type Parameters = faer::Col<T>;
    // we always return the owned type here due to implementation details in the
    // dogleg crate.
    type Jacobian = faer::Mat<T>;

    fn set_params(&mut self, x: Self::Parameters) {
        self.inner.set_params(&nalgebra_vec_from_faer_col(x));
    }

    fn params(&self) -> Self::Parameters {
        faer_col_from_nalgebra_vector(self.inner.params())
    }

    fn residuals(&self) -> Option<Self::Residuals> {
        Some(faer_col_from_nalgebra_vector(self.inner.residuals()?))
    }

    fn jacobian(&self) -> Option<Self::Jacobian> {
        Some(faer_mat_from_nalgebra_matrix(self.inner.jacobian()?))
    }
}

fn nalgebra_vec_from_faer_col<T, R>(col: faer::Col<T>) -> nalgebra::OVector<T, R>
where
    T: Scalar + Zero,
    nalgebra::DefaultAllocator: Allocator<R>,
    R: nalgebra::Dim,
{
    // NOTE PERF also a very dumb impl, but it's easy to check that it's correct
    let mut v = OVector::<T, R>::zeros_generic(R::from_usize(col.nrows()), U1);

    v.iter_mut()
        .zip(col.iter())
        .for_each(|(v, c)| *v = c.clone());
    v
}

fn faer_mat_from_nalgebra_matrix<T, R, C, S>(mat: nalgebra::Matrix<T, R, C, S>) -> faer::Mat<T>
where
    T: Copy + nalgebra::ComplexField + Float + faer::traits::RealField,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: RawStorage<T, R, C>,
{
    // the faer-ext crate isn't kept in sync with the latest faer release,
    // so we'll hack something that's way less efficient here and make this
    // owned matrices.
    //
    // WARN PERF This is very stupid code, but it's easy to see
    // that this is correct, which is what I want for testing.
    faer::Mat::from_fn(mat.nrows(), mat.ncols(), |i, j| mat[(i, j)])
}

fn faer_col_from_nalgebra_vector<T, R, S>(v: nalgebra::Vector<T, R, S>) -> faer::Col<T>
where
    R: nalgebra::Dim,
    T: Copy + nalgebra::ComplexField + Float + faer::traits::RealField,
    S: RawStorage<T, R>,
{
    // WARN PERF see above, very stupid but correct
    faer::Col::from_fn(v.nrows(), |i| v[(i, 0)])
}
