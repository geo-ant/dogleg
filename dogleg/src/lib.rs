//! # dogleg
//!
//! > _"Obviously you're not a golfer"_
//! >
//! > -- Jeffrey "The Dude" Lebowski
//!
//! This crate implements [Powell's Dogleg Algorithm](https://en.wikipedia.org/wiki/Powell%27s_dog_leg_method)
//! to solve a least squares problem of the form:
//!
//! ```math
//! \min_{x} \frac{1}{2}\lVert\boldsymbol{r}(\boldsymbol{x})\rVert_2^2,
//! ```
//!
//! where `$\boldsymbol{r}\in\mathbb{R}^m$` is called the _residual(s)_,
//! `$\boldsymbol{x}\in\mathbb{R}^n$` is called the _parameter(s)_. We typically
//! call `$f(\boldsymbol{x}):=\frac{1}{2}\lVert\boldsymbol{r}(\boldsymbol{x})\rVert_2^2$`
//! the _objective function_ to minimize. Finally, and this will surprise nobody,
//! `$\lVert.\rVert_2$` is the [Euclidean norm](https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm).
//!
//! To use the algorithm in the dogleg crate, you'll need to be able to calculate
//! both the residual vector `$\boldsymbol{r}$`, as well as its
//! [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)
//! `$\boldsymbol{J}$` for a given parameter `$\boldsymbol{x}$`.
//! The Jacobian is defined as
//!
//! ```math
//! \boldsymbol{J}=
//! \left(\begin{matrix}
//! \nabla^T r_1(\boldsymbol{x}) \\[0.75em]
//! \vdots  \\[0.75em]
//! \nabla^T r_m(\boldsymbol{x}) \\
//! \end{matrix}\right)
//! =
//! \left(\begin{matrix}
//! \frac{\partial}{\partial x_1} r_1(\boldsymbol{x}) & \dots & \frac{\partial}{\partial x_n} r_1(\boldsymbol{x})  \\[0.75em]
//! \frac{\partial}{\partial x_1} r_2(\boldsymbol{x}) & \dots & \frac{\partial}{\partial x_n} r_2(\boldsymbol{x})  \\
//! \vdots & \ddots & \vdots \\
//! \frac{\partial}{\partial x_1} r_m(\boldsymbol{x}) & \dots & \frac{\partial}{\partial x_n} r_m(\boldsymbol{x})  \\
//! \end{matrix}\right)
//! ```
//!
//! ## Example
//!
//! Let's say you want to minimize the following residual `$\boldsymbol{r}$`,
//! which depends on parameters `$x_1$` and `$x_2$`, which form the parameter
//! vector `$\boldsymbol{x} = (x_1,x_2)^T$`:
//!
//! ```math
//! \boldsymbol{r}(\boldsymbol{x}) =
//! \left(\begin{matrix}
//! r_1(\boldsymbol{x}) \\[0.75em]
//! r_2(\boldsymbol{x})
//! \end{matrix}\right)
//! =
//! \left(\begin{matrix}
//! x_1 \, x_2 -1 \\[0.75em]
//! (x_1 - 1)^2 + (x_2 -1)^2
//! \end{matrix}\right)
//! ```
//!
//! That means the Jacobian is:
//!
//! ```math
//! \boldsymbol{J}=
//! \left(\begin{matrix}
//! \frac{\partial}{\partial x_1} r_1(\boldsymbol{x}) & \frac{\partial}{\partial x_2} r_1(\boldsymbol{x})  \\[0.75em]
//! \frac{\partial}{\partial x_1} r_2(\boldsymbol{x}) & \frac{\partial}{\partial x_2} r_2(\boldsymbol{x})  \\
//! \end{matrix}\right)
//! =
//! \left(\begin{matrix}
//! x_2       & x_1 \\[0.75em]
//! 2 (x_1-1) & 2 (x_2 -1) \\        
//! \end{matrix}\right)
//! ```
//!
//! For the implementation, we assume you're using the default linear algebra
//! backend, which is [`nalgebra`](https://crates.io/crates/nalgebra). That's why
//! we'll express our matrices and vectors using the types from the `nalgebra`
//! crate. If you prefer a different linear algebra backend, please see the
//! section on choosing a linear algebra backend.
//!
//! ```rust
#![doc = include_str!("../examples/simple.rs")]
//! ```
//!
//! ### Optimizing Your Problem Implementation for Computational Efficiency
//!
//! In the example above, the only state that we've kept in the problem are
//! the parameters themselves, but we can do more than that.
//! Note, that [`LeastSquaresProblem::set_params`] is the only trait method that takes
//! self by _mutable_ (exclusive) reference. That means any call to calculate
//! the residuals or Jacobian uses those same parameters. That enables us to
//! perform some shared calculations during `set_params` and reuse them
//! in both the residual and the Jacobian calculations.
//!
//! This can significantly speed up the solver if you pay attention to the
//! following details:
//!
//! 1. Always, always, always _measure_ when trying to optimize performance!
//! 2. If at all possible, only share calculations that are reused by _both_ the residuals
//!   _and_ the Jacobian. All other calculations should go into the respective
//!   methods.
//!
//! If you're having trouble figuring out _which_ shared calculations to factor out,
//! consider this: the residuals will be calculated more often, typically _much_
//! more often, than the Jacobian itself. Take that into account when
//! deciding what to cache.
//!
//! ## Choosing A Linear Algebra Backend
//!
//! By default this crate comes with [`nalgebra`](https://crates.io/crates/nalgebra)
//! as the matrix backend. This means
//!
//!
//!
//! ## Goals And Non-Goals
//!
//! This crate has the following **goals**:
//!
//! * [x] Rust-native least-squares minimization with a simple, yet powerful interface.
//! * [x] Support different linear algebra backends. Contributions very welcome.
//!   * [x] `faer-rs`
//!   * [x] `nalgebra`
//!   * [ ] `ndarray`
//! * [x] Configurable solver parameters and algorithm details
//! * [x] Numerical performance on par with Ceres Solver (for the set of test problems)
//! * [x] _Almost_ drop-in replacement for the solver from the `levenberg-marquardt`
//!   crate. Please see the section on compatibility.
//! * [ ] High performance (fast) minimization.
//!   * [ ] Fewer allocations internally
//!   * [ ] Faster matrix decompositions
//!   * [ ] Some algebraic optimizations
//! * [ ] `no_std`. This one should be easy, I just haven't explicitly done it yet.
//!
//! However, there are things that this crate **is not** and will never be:
//!
//! * A minimization _framework_. `dogleg` is only ever supposed to do one thing
//!   well, which is least squares minimization using the Dogleg Algorithm.
//!   If you are looking for an optimization framework, try e.g.
//!   [`argmin`](https://crates.io/crates/argmin).
//! * _Safe_ Rust _only_. The `dogleg` crate itself will probably not have
//!   to use unsafe code outright, but the downstream `dogleg-matx` does use
//!   it (sparingly) and the downstream linear algebra crates use it, too.
//!   If someone were willing to implement all the linear algebra abstractions
//!   using only safe Rust all the way to the bottom, I'd be willing to
//!   commit myself to `dogleg` itself not using `unsafe`.
//!
//! ## (Almost) Drop-in Compatibility with `levenberg-marquardt`
//!
//! Since this crate is heavily inspired by the
//! [`levenberg-marquardt`](https://crates.io/crates/levenberg-marquardt) crate,
//! I thought it'd be nice to have a low entry barrier
//! for folks already using that crate. In that case, you've already implemented
//! [`levenberg-marquardt::LeastSquaresProblem`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/trait.LeastSquaresProblem.html)
//! and you can just use [`LevMarAdapter`] to turn
//! that into a [`dogleg::LeastSquaresProblem`](crate::LeastSquaresProblem) that can
//! be minimized with a [`Dogleg`] instance.
//!
//! The `Dogleg` and `LevenbergMarquardt` interface is very similar, so there's
//! a good chance you can just replace `LevenbergMarquardt` with `Dogleg`
//! and be almost good to go. However, `dogleg` uses a `Result`-based approach
//! to communicate success or failure of the minimization, whereas
//! the `levenberg-marquardt` crate wants you to check the termination reason
//! manually. Still, it should be very easy to plug in `dogleg` and just try
//! it out.
//!
//! ## Acknowledgments & References
//!
//! This crate wouldn't exist without many other projects and resources for
//! which I'm very grateful:
//!
//! * [`levenberg-marquardt`](https://crates.io/crates/levenberg-marquardt):
//!   The least squares problem interface as well as the set of classic MGH
//!   test problems were taken from this excellent crate.
//! * [`Ceres Solver`](https://github.com/ceres-solver/ceres-solver): This
//!   was the solver to beat and I ended up taking a lot of algorithmic
//!   implementation details from it.
//! * The [`argmin`](https://crates.io/crates/argmin) crate: this is, as far
//!   as I know, the first numerical optimization crate for Rust that's truly agnostic
//!   over linear algebra backends.
//! * [`MINPACK`](https://netlib.org/minpack/), the godfather of least squares
//!   minimization libraries and the much more readable
//!   [`Modern MINPACK`](https://github.com/fortran-lang/minpack) port. The
//!   dogleg algorithm in this crate started out as an adaptation of the MINPACK algorithm,
//!   but moved away from it over time. Still, there's a decent amount of
//!   MINPACK-inspired code in this crate.
//! * The book [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5)
//!   _2nd ed._ by Nocedal&Wright as well as the paper
//!   [Methods for Non-Linear Least Squares Problems](https://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf)
//!   by Madsen, Nielsen, and Tingleff.
//!
//! ## Contributing
//!
//! Contributions are very welcome. If you submit a contribution, you'll have to
//! do so under the license that this project is under. That implies that you must
//! have the right to do so, which won't be a problem if you're doing
//! this in your free time.
//!
//! ### Generative AI Policy
//!
//! This project follows the [brainmade.org](https://brainmade.org/) idea of
//! AI usage. Quoting:
//!
//! > _"There’s something transcendent and magical in knowing a human made the_
//! > _artwork I’m consuming, knowing they tried hard is part of the experience._
//! > _It doesn’t have to be 100% human made (what would that even MEAN these_
//! > _days?), perhaps 90% human made."_
//!
//! There are some examples and clarifications on that page, which I find useful.
//! There's one sentence I disagree with, but that's a difference in opinion between
//! me and the valued author of that page. That sentence is _"I’m [...] a software developer,
//! I’ll re-use 100 libraries to avoid writing 10 lines of code"_. We don't do
//! that here.
//!
//! ## What on Earth Does "Obviously You're Not a Golfer" Mean??
//!
//! Have you, like me, ever wondered where the Dogleg Algorithm got its name?
//! Apparently its inventor [Michael JD Powell](https://en.wikipedia.org/wiki/Michael_J._D._Powell),
//! unlike me, was an avid golfer. Turns out, you can visualize the step construction in the algorithm
//! such that it looks like a [_dogleg hole_](https://en.wikipedia.org/wiki/Golf_course#Fairway_and_rough)
//! in golf. The illustration of the dogleg step in Wikipedia is actually not
//! very helpful to see this similarity, but the illustation in Nocedal&Wright
//! _is_. And then there's [The Big Lebowski](https://en.wikipedia.org/wiki/The_Big_Lebowski),
//! a cult classic, from which the quote is taken.

#![deny(unsafe_code)]
//@todo(geo) reinstate thsi
#![warn(missing_docs)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::panicking_unwrap)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(test)]
mod test;

#[cfg(test)]
mod test_adapters;

/// solver implementations
pub mod dogleg;
/// error types
mod error;
/// least squares problem abstractions and levmar compatibility
mod problem;

pub use dogleg::report::TerminationFailure;
pub use dogleg::report::TerminationReason;
pub use dogleg::Dogleg;
pub use problem::LeastSquaresProblem;

pub use dogleg_matx as matx;
pub use dogleg_matx::magic_const::MagicConst;
pub use error::Error;

/// re-export the levenberg-marquardt crate
#[cfg(feature = "levenberg-marquardt")]
pub use levenberg_marquardt;
#[cfg(feature = "levenberg-marquardt")]
pub use problem::levmar_adapter::LevMarAdapter;
