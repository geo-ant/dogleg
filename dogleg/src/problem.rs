use dogleg_matx::{Colx, Matx, OwnedColx};

/// compatibility type for the levenberg-marquardt crate
#[cfg(feature = "levenberg-marquardt")]
pub mod levmar_adapter;

/// Describes the least squares problem to minimize. See the [top level docs](index.html)
/// for usage.
pub trait LeastSquaresProblem<T> {
    /// column vector of size `$m$` for the residuals.
    type Residuals: Colx<T> + std::fmt::Debug;
    /// column vector of size `$n$` for the parameters.
    /// We require the parameter vector to be an owned vector which is slightly
    /// more restrictive than the `levenberg-marquardt` crate
    type Parameters: OwnedColx<T> + Clone + std::fmt::Debug;
    /// matrix of size `$m \times n` for the Jacobian of the residuals
    type Jacobian: Matx<T> + std::fmt::Debug;

    /// Set the stored parameters `$\boldsymbol{x}$`.
    fn set_params(&mut self, x: Self::Parameters);

    /// Get the current parameters
    fn params(&self) -> Self::Parameters;

    /// Compute the residual vector at the current parameters.
    /// Return None to communicate failure.
    fn residuals(&self) -> Option<Self::Residuals>;

    /// Compute the Jacobian of the residual vector at the current parameters.
    /// Return None to communicate failure.
    fn jacobian(&self) -> Option<Self::Jacobian>;
}
