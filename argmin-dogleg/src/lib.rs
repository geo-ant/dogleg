//! utility crate that provides a way to use argmin on
//! `levenberg-marquardt::LeastSquaresProblem`

mod levmar_adapter;

pub use levmar_adapter::run_argmin_dogleg;
pub use levmar_adapter::ArgminLevMarAdapter;
