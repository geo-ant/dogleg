use dogleg_matx::{Colx, Matx};
use nalgebra::{
    allocator::Allocator, ComplexField, DefaultAllocator, Dim, IsContiguous, Matrix, RawStorageMut,
    RealField, Storage, Vector,
};

use crate::sealed::Sealed;

/// Describes a least squares minimization problem.
///
/// This trait is compatible with the
/// [levenberg-marquardt crate](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/trait.LeastSquaresProblem.html),
/// as a matter of fact, it's directly taken from it.
pub trait LevMarLeastSquaresProblem<F, M, N>
where
    F: ComplexField + Copy,
    N: Dim,
    M: Dim,
{
    /// Storage type used for the residuals. Use `nalgebra::storage::Owned<F, M>`
    /// if you want to use `VectorN` or `MatrixMN`.
    type ResidualStorage: RawStorageMut<F, M> + Storage<F, M> + IsContiguous;
    type JacobianStorage: RawStorageMut<F, M, N> + Storage<F, M, N> + IsContiguous;
    type ParameterStorage: RawStorageMut<F, N> + Storage<F, N> + IsContiguous + Clone;

    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, x: &Vector<F, N, Self::ParameterStorage>);

    /// Get the current parameter vector `$\vec{x}$`.
    fn params(&self) -> Vector<F, N, Self::ParameterStorage>;

    /// Compute the residual vector.
    fn residuals(&self) -> Option<Vector<F, M, Self::ResidualStorage>>;

    /// Compute the Jacobian of the residual vector.
    fn jacobian(&self) -> Option<Matrix<F, M, N, Self::JacobianStorage>>;
}

// TODO update docs
// @note(geo) this is the trait that dependent crate should implement
pub trait LeastSquaresProblem<T> {
    /// column vector of size M for the residuals
    type Residuals: Colx<T>;
    /// column vector of size N for the parameters
    type Parameters: Colx<T>;
    /// matrix of size M x N for the Jacobian of the residuals
    type Jacobian: Matx<T>;

    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, x: Self::Parameters);

    /// Get the current parameter vector `$\vec{x}$`.
    fn params(&self) -> Self::Parameters;

    /// Compute the residual vector.
    fn residuals(&self) -> Option<Self::Residuals>;

    /// Compute the Jacobian of the residual vector.
    fn jacobian(&self) -> Option<Self::Jacobian>;
}

impl<S, T> Sealed for S where S: LeastSquaresProblem<T> {}

impl<T, P> LeastSquaresProblemSealed<T, ()> for P
where
    P: LeastSquaresProblem<T>,
{
    type Residuals = <P as LeastSquaresProblem<T>>::Residuals;
    type Parameters = <P as LeastSquaresProblem<T>>::Parameters;
    type Jacobian = <P as LeastSquaresProblem<T>>::Jacobian;

    fn set_params(&mut self, x: Self::Parameters) {
        LeastSquaresProblem::<T>::set_params(self, x)
    }

    fn params(&self) -> Self::Parameters {
        LeastSquaresProblem::<T>::params(self)
    }

    fn residuals(&self) -> Option<Self::Residuals> {
        LeastSquaresProblem::<T>::residuals(self)
    }

    fn jacobian(&self) -> Option<Self::Jacobian> {
        LeastSquaresProblem::<T>::jacobian(self)
    }
}

//TODO this is just so that we can also implement our internal
// trait for levenberg-marquardt crate compatibility
pub trait LeastSquaresProblemSealed<T, Dummy>: Sealed {
    /// column vector of size M for the residuals
    type Residuals: Colx<T>;
    /// column vector of size N for the parameters
    type Parameters: Colx<T>;
    /// matrix of size M x N for the Jacobian of the residuals
    type Jacobian: Matx<T>;

    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, x: Self::Parameters);

    /// Get the current parameter vector `$\vec{x}$`.
    fn params(&self) -> Self::Parameters;

    /// Compute the residual vector.
    fn residuals(&self) -> Option<Self::Residuals>;

    /// Compute the Jacobian of the residual vector.
    fn jacobian(&self) -> Option<Self::Jacobian>;
}

impl<T, P, M, N> LeastSquaresProblemSealed<T, (M, N)> for P
where
    T: ComplexField + Copy + RealField,
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
        <P as LevMarLeastSquaresProblem<_, _, _>>::set_params(self, &x);
    }

    fn params(&self) -> Self::Parameters {
        <P as LevMarLeastSquaresProblem<_, _, _>>::params(self)
    }

    fn residuals(&self) -> Option<Self::Residuals> {
        <P as LevMarLeastSquaresProblem<_, _, _>>::residuals(self)
    }

    fn jacobian(&self) -> Option<Self::Jacobian> {
        <P as LevMarLeastSquaresProblem<_, _, _>>::jacobian(self)
    }
}
