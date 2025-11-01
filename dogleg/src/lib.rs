//@todo(geo) reinstate thsi
// #![warn(missing_docs)]
//@todo(geo) remove this
#![allow(dead_code)]

/// solver implementations
pub mod dogleg;
/// error types
mod error;
/// utility
mod magic_const;
/// least squares problem abstractions and levmar compatibility
mod problem;

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
