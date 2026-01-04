//! utility crate that provides a way to use argmin on
//! `levenberg-marquardt::LeastSquaresProblem`

mod levmar_adapter;

#[cfg(test)]
mod tests;

pub use levmar_adapter::run_argmin_dogleg as argmin_solve_with_dogleg;
pub use levmar_adapter::ArgminLevMarAdapter;
pub use levmar_adapter::ArgminReport;
