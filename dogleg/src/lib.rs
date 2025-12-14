//@todo(geo) reinstate thsi
// #![warn(missing_docs)]
//@todo(geo) remove this
// #![allow(dead_code)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

/// solver implementations
pub mod dogleg;
/// error types
mod error;
/// utility
mod magic_const;
/// least squares problem abstractions and levmar compatibility
mod problem;

pub use dogleg::report::TerminationFailure;
pub use dogleg::report::TerminationReason;
pub use dogleg::Dogleg;
pub use problem::LeastSquaresProblem;

pub use dogleg_matx as matx;
pub use error::Error;
pub use magic_const::MagicConst;

/// re-export the levenberg-marquardt crate
#[cfg(feature = "levenberg-marquardt")]
pub use levenberg_marquardt;
#[cfg(feature = "levenberg-marquardt")]
pub use problem::levmar_adapter::LevMarAdapter;
