use dogleg_matx::{Colx, Matx, OwnedColx};

/// compatibility type for the levenberg-marquardt crate
#[cfg(feature = "levenberg-marquardt")]
pub mod levmar_adapter;

//TODO document
pub trait LeastSquaresProblem<T> {
    /// column vector of size M for the residuals
    type Residuals: Colx<T>;
    /// column vector of size N for the parameters.
    /// We require the parameter vector to be an owned vector which is slightly
    /// more restrictive than the `levenberg-marquardt` crate
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
