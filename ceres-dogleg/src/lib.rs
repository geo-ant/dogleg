//! this crate needs ceres solver installed. The installation itself via
//! building from source should be easy, but I ran into tons of linker problems.
//! What eventually fixed it was
//!
//! ```shell
//! # apt install libceres-dev
//! ```
use anyhow::{anyhow, bail};
use ceres_solver::{
    CostFunctionType, NllsProblem,
    solver::{LinearSolverType, LoggingType, MinimizerType},
};
use core::f64;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{DefaultAllocator, Dim, allocator::Allocator};
use std::sync::{Arc, Mutex, Once};

pub use ceres_solver::SolverOptions;
extern crate ceres_solver;

pub struct CeresReport {
    pub objective_function: f64,
}

#[cfg(test)]
mod test;

// binding for cpp/glogging.cpp
unsafe extern "C" {
    fn init_glog_for_ceres(verbosity: i32);
}

static GLOGGING_INIT: Once = Once::new();

/// given options.
pub fn ceres_solve_with_dogleg<P, M, N>(problem: P) -> anyhow::Result<(P, CeresReport)>
where
    M: Dim,
    N: Dim,
    P: LeastSquaresProblem<f64, M, N>,
    P::JacobianStorage: Clone,
    DefaultAllocator: Allocator<N> + Allocator<M> + Allocator<N, M>,
{
    GLOGGING_INIT.call_once(|| unsafe { init_glog_for_ceres(2) });
    let options = SolverOptions::builder()
        .minimizer_type(MinimizerType::TRUST_REGION)
        .trust_region_strategy_type(ceres_solver::solver::TrustRegionStrategyType::DOGLEG)
        .linear_solver_type(LinearSolverType::DENSE_QR)
        // NOTE: could be helpful for some high level logging, but does NOT
        // give us the VLOG(n) output.
        .update_state_every_iteration(true)
        // .minimizer_progress_to_stdout(true)
        .logging_type(LoggingType::PER_MINIMIZER_ITERATION)
        .build()
        .unwrap();
    ceres_solve_with_options(problem, options)
}

/// use the ceres solver to solve the given least squares problem with the
/// given options.
fn ceres_solve_with_options<P, M, N>(
    problem: P,
    options: SolverOptions,
) -> anyhow::Result<(P, CeresReport)>
where
    M: Dim,
    N: Dim,
    P: LeastSquaresProblem<f64, M, N>,
    P::JacobianStorage: Clone,
    DefaultAllocator: Allocator<N> + Allocator<M> + Allocator<N, M>,
{
    let residual_dim = problem
        .residuals()
        .ok_or(anyhow!("could not calculate residuals"))?
        .len();
    let mut initial_params = problem.params();

    // hack because the close has type Fn, not FnMut
    let problem = Arc::new(Mutex::new(problem));
    let cost: CostFunctionType = Box::new(|parameters, residuals, jacobian| {
        let problem = &problem;
        // see http://ceres-solver.org/nnls_modeling.html
        assert_eq!(parameters.len(), 1);
        let parameters = parameters[0]; // we expect only a single parameter block

        // I think this is the correct way to get the jacobian for the one parameter block
        // NOTE(geo-ant): the jacobian is ROW-MAJOR for ceres!
        let jacobian_rowmajor_array = match jacobian {
            Some(slice) => {
                assert_eq!(
                    slice.len(),
                    1,
                    "jacobian must only have one parameter block"
                );
                match &mut slice[0] {
                    Some(opt) => Some(opt),
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

        if let Some(jacobian) = jacobian_rowmajor_array {
            let jac_tr = problem_guard.jacobian().unwrap().transpose();
            // the ceres jacobian is in row major order
            // so we write the transposed jacobian to
            // the provided memory

            assert_eq!(jacobian.len(), jac_tr.ncols());
            assert_eq!(jacobian[0].len(), jac_tr.nrows());

            // nalgebra view storages are complete crap, so this won't compile
            // using iterators. Who knows, maybe I'm doing something wrong...
            for idc in 0..jac_tr.ncols() {
                for idr in 0..jac_tr.nrows() {
                    jacobian[idc][idr] = jac_tr[(idr, idc)];
                }
            }
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
    let best_params = initial_params;

    // just to make sure that the problem has indeed currently set the best
    // params.
    let mut problem = Arc::into_inner(problem).unwrap().into_inner().unwrap();
    problem.set_params(&best_params);

    if !solution.summary.is_solution_usable() {
        bail!("CERES solver indicates solution is not usable");
    }

    // just check that no inner iteration steps are performed
    assert_eq!(solution.summary.num_inner_iteration_steps(), -1);

    Ok((
        problem,
        CeresReport {
            objective_function: solution.summary.final_cost(),
        },
    ))
}
