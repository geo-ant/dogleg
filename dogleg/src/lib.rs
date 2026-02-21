//@todo(geo) reinstate thsi
// #![warn(missing_docs)]
//@todo(geo) remove this
// #![allow(dead_code)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

#[cfg(test)]
mod test;

/// solver implementations
pub mod dogleg;
/// error types
mod error;
/// least squares problem abstractions and levmar compatibility
mod problem;

pub use dogleg::report::TerminationFailure;
pub use dogleg::report::TerminationReason;
pub use dogleg::Dogleg;
pub use problem::LeastSquaresProblem;

pub use dogleg_matx as matx;
pub use dogleg_matx::magic_const::MagicConst;
pub use error::Error;

/// re-export the levenberg-marquardt crate
#[cfg(feature = "levenberg-marquardt")]
pub use levenberg_marquardt;
#[cfg(feature = "levenberg-marquardt")]
pub use problem::levmar_adapter::LevMarAdapter;
