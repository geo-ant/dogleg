use crate::{dogleg::MinimizationReport, LeastSquaresProblem};
use dogleg_matx::{magic_const::MagicConst, Colx};
use nalgebra::RealField;
use num_traits::Float;

use crate::Error;

/// report that captures the most important info from both error and success reports
pub struct DebugReport<T> {
    pub objective_function: T,
    pub success: bool,
}

impl<T> From<MinimizationReport<T>> for DebugReport<T> {
    fn from(value: MinimizationReport<T>) -> Self {
        Self {
            objective_function: value.objective_function,
            success: true,
        }
    }
}

/// useful for debugging to have a unified interface
pub trait IntoDebugReport<P, T> {
    fn into_debug_report(self) -> (P, DebugReport<T>);
}

impl<P, T> IntoDebugReport<P, T> for Result<(P, MinimizationReport<T>), Error<P>>
where
    T: Float + MagicConst,
    P: LeastSquaresProblem<T>,
{
    fn into_debug_report(self) -> (P, DebugReport<T>) {
        match self {
            Ok((p, r)) => (p, r.into()),
            Err(err) => {
                let objective_function =
                    T::P5 * Float::powi(err.problem.residuals().unwrap().enorm(), 2);
                (
                    err.problem,
                    DebugReport {
                        objective_function,
                        success: false,
                    },
                )
            }
        }
    }
}
