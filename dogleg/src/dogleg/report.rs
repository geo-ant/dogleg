//! this module contains a decent amount of code directly copied from the
//! `levenberg-marquardt` crate. Some code is more or less lightly modified.
//! The original license for this code is as such:
//!
//! MIT License
//!
//! Copyright (c) 2020 rust-cv
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.

//@todo document
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StoppingCriterion {
    Xtol,
    Ftol,
    Gtol,
}

#[derive(Debug, Clone, PartialEq)]
/// Information about the minimization.
///
/// Use this to inspect the minimization process. Most importantly
/// you may want to check if there was a failure.
pub struct MinimizationReport<T> {
    pub termination: TerminationReason,
    /// Number of residuals which were computed.
    pub number_of_evaluations: u64,
    /// Contains the value of `f(x)=1/2 ||r(x)||^2`.
    pub objective_function: T,
}

#[derive(PartialEq, Eq, Debug, Clone)]
/// Reason for a successful termination of the minimization.
pub enum TerminationReason {
    /// The residuals are literally zero.
    ResidualsZero,
    /// Convergence achieved. Also tells us which stopping criterion
    /// triggered the end of the iteration. This doesn't necessarily
    /// mean that this is the _only_ criterion that would be hit in
    /// the last iteration, it just means the others weren't evaluated
    /// anymore.
    Converged(StoppingCriterion),
}

#[derive(PartialEq, Eq, Debug, Clone)]
/// Reason for failure of the minimization.
pub enum TerminationFailure {
    /// Encountered `NaN` or `+/- inf`.
    Numerical(&'static str),
    /// Jacobian evaluation failed
    JacobianEval,
    /// residual evaluation failed
    ResidualEval,
    /// The bound for `ftol`, `xtol` or `gtol` was set so low that the
    /// test passed with the machine epsilon but not with the actual
    /// bound. This means you must increase the bound.
    NoImprovementPossible(StoppingCriterion),
    /// Maximum number of function evaluations was hit.
    LostPatience,
    /// Wrong dimensions for a matrix or vector
    WrongDimensions(&'static str),
    /// Matrix dimensions are too large
    DimOutsideU64Bounds,
}
