use core::f64;
use std::sync::Mutex;

use ceres_solver::{CostFunctionType, types::JacobianType};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{Dim, RealField, SliceStorageMut};

pub fn solve_dogleg<P, M, N>(mut problem: P)
where
    M: Dim,
    N: Dim,
    P: LeastSquaresProblem<f64, M, N>,
{
    // hack because the close has type Fn, not FnMut
    let mut problem = Mutex::new(problem);
    let cost: CostFunctionType = Box::new(move |parameters, residuals, jacobian| {
        // see http://ceres-solver.org/nnls_modeling.html
        assert_eq!(parameters.len(), 1);
        let parameters = parameters[0]; // we expect only a single parameter block

        // I think this is the correct way to get the jacobian for the one parameter block
        // NOTE(geo-ant): the jacobian is ROW-MAJOR for ceres!
        let jacobian: Option<&mut [f64]> = match jacobian {
            Some(slice) => {
                assert_eq!(slice.len(), 1);
                match &mut slice[0] {
                    Some(opt) => Some(opt[0]),
                    None => None,
                }
            }
            None => None,
        };
        let mut problem_guard = problem.lock().unwrap();

        let mut p = problem_guard.params();
        if p.len() != parameters.len() {
            panic!(
                "wrong parameter length: expected {}, got {}",
                p.len(),
                parameters.len()
            );
        }
        p.copy_from_slice(parameters);
        problem_guard.set_params(&p);

        let res = problem_guard.residuals().unwrap();
        assert_eq!(
            res.len(),
            residuals.len(),
            "wrong length of residuals: expected {}, got {}",
            res.len(),
            residuals.len()
        );
        residuals.copy_from_slice(res.as_slice());

        if let Some(jacobian) = jacobian {
            let jac = problem_guard.jacobian();
            todo!("set jacobian in ROW major");
        }

        todo!()
    });
}
