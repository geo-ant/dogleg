//! this crate needs ceres solver installed, which is very easy to do by
//! downloading the release package and building and installing it via
//! cmake.

use anyhow::anyhow;
use ceres_solver::{CostFunctionType, NllsProblem};
use core::f64;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{DefaultAllocator, Dim, Vector, allocator::Allocator};
use std::sync::Mutex;

pub use ceres_solver::SolverOptions;
extern crate ceres_solver;

/// use the ceres solver to solve the given least squares problem with the
/// given options.
pub fn ceres_solve_with_options<P, M, N>(
    problem: P,
    options: SolverOptions,
) -> anyhow::Result<Vector<f64, N, P::ParameterStorage>>
where
    M: Dim,
    N: Dim,
    P: LeastSquaresProblem<f64, M, N>,
    DefaultAllocator: Allocator<N>,
{
    let residual_dim = problem
        .residuals()
        .ok_or(anyhow!("could not calculate residuals"))?
        .len();
    let mut initial_params = problem.params();

    // hack because the close has type Fn, not FnMut
    let problem = Mutex::new(problem);
    // cost function allows us to provide residuals and jacobian to the ceres solver
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

        // set the parameters. This is a bit hacky, but getting the
        // parameters, overwriting, and then setting them is an easy
        // way of obtaining a parameter vector of the correct type.
        let mut p = problem_guard.params();
        assert_eq!(
            p.len(),
            parameters.len(),
            "wrong parameter length: expected {}, got {}",
            p.len(),
            parameters.len()
        );
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
            let jac = problem_guard.jacobian().unwrap();
            // the ceres jacobian is in row major order
            // so we write the transposed jacobian to
            // the provided memory
            assert_eq!(
                jacobian.len(),
                jac.nrows() * jac.ncols(),
                "wrong jacobian dimensions, expected {}, got {}",
                jac.nrows() * jac.ncols(),
                jacobian.len()
            );
            jacobian.copy_from_slice(jac.as_slice());
        }
        true
    });

    // see https://docs.rs/ceres-solver/0.5.0/ceres_solver/index.html#reexport.CostFunctionType
    // the interface is a bit awkward...
    let solution = NllsProblem::new()
        .residual_block_builder()
        .set_cost(cost, residual_dim)
        // the interface is a bit weird, but we only have one parameter
        // block, so this should do the trick.
        .set_parameters(vec![initial_params.as_slice().to_vec()])
        .build_into_problem()?
        .0
        .solve(&options)?;
    let best_params = solution.parameters;
    assert_eq!(
        best_params.len(),
        1,
        "expected only one parameter block, got {}",
        best_params.len()
    );
    let best_params = best_params.into_iter().next().unwrap();

    assert_eq!(
        best_params.len(),
        initial_params.len(),
        "unexpected parameter len: expected {}, got {}",
        initial_params.len(),
        best_params.len()
    );

    // we overwrite the initial param vector and return it.
    // This is just an easy way of getting an instance of the correct
    // type of parameter vector.
    initial_params.copy_from_slice(&best_params);

    Ok(initial_params)
}
