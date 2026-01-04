//! utility crate that provides a way to use argmin on
//! `levenberg-marquardt::LeastSquaresProblem`

mod levmar_adapter;

#[cfg(test)]
mod test;

pub use levmar_adapter::argmin_solve_with_dogleg;
pub use levmar_adapter::ArgminLevMarAdapter;
pub use levmar_adapter::ArgminReport;
