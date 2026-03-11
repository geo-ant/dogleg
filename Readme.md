# dogleg 

**Backend-agnostic high-quality Rust-native unconstrained least-squares minimization**

![build](https://github.com/geo-ant/dogleg/actions/workflows/build.yml/badge.svg?branch=main)
![tests](https://github.com/geo-ant/dogleg/actions/workflows/tests.yml/badge.svg?branch=main)
![lints](https://github.com/geo-ant/dogleg/actions/workflows/lints.yml/badge.svg?branch=main)
[![crates](https://img.shields.io/crates/v/dogleg)](https://crates.io/crates/dogleg)
[![Coverage Status](https://coveralls.io/repos/github/geo-ant/dogleg/badge.svg?branch=main)](https://coveralls.io/github/geo-ant/dogleg?branch=main)
![maintenance-status](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)
[![ko-fi](https://raw.githubusercontent.com/geo-ant/user-content/refs/heads/main/ko-fi-support.svg)](https://ko-fi.com/geoant)

> _"Obviously you're not a golfer"_
>
> -- Jeffrey "The Dude" Lebowski

Solve unconstrained least-squares problems using [Powell's Dogleg](https://en.wikipedia.org/wiki/Powell%27s_dog_leg_method)
algorithm. For the problems in the test suite, this implementation is on
par with Google's [Ceres Solver](http://ceres-solver.org/) and the 
[`levenberg-marquardt`](https://crates.io/crates/levenberg-marquardt) crate
in terms of _numerical quality_ of the results, but not yet speed. It performs
much better than the [`argmin-rs`](https://argmin-rs.org/) implementation of
the Dogleg Algorithm on the same problems.

## Usage

Please see the [documentation](https://docs.rs/dogleg/) for detailed explanations
and examples. If you know the excellent [`levenberg-marquardt`](https://crates.io/crates/levenberg-marquardt)
crate, you'll notice that the interface is very similar. This is on purpose.

```rust
impl LeastSquaresProblem<f64> for Problem {
    // describes the problem using the residuals
    // and the Jacobian (derivatives)
}

let problem = Problem::new(initial_params);
let (problem, report) = Dogleg::new().minimize(problem).unwrap();
let minimum = problem.params();
```

## Why Dogleg?

There are two parts to this answer, why did I write this and why would you use
it? I've relegated the answer to the first question [to the docs](https://docs.rs/dogleg/).
So why might you want to use it?

* **High Numerical Quality**: The quality of the numerical results is on par
  with the Rust-native `levenberg-marquardt` crate and the Ceres Solver
  implementation of the Dogleg Algorithm.
* **Backend Agnostic**: This crate does not lock you in to a specific linear
  algebra backend. For now, both [`nalgebra`](https://crates.io/crates/nalgebra)
  and [`faer-rs`](https://crates.io/crates/faer) are supported, but implementing
  the necessary abstractions on your own backend is possible.

Please note that this crate is an MVP. I've painstakingly optimized the numerical
quality of the results, but I haven't yet optimized for solver speed. Expect
significant speedups in the next releases.

## More Information

I've written answers to all of the following questions [in the docs](https://docs.rs/dogleg/),
so please check them out if you're curious:

* Can I see an actual example?
* How do I choose a linear algebra backend?
* What are the goals and non-goals of this crate?
* Can I use this as a drop-in replacement for `levenberg-marquardt`?
* Why did you write this crate?
* What other crates did inspire you?
* How can I contribute?
* Did you use AI to write this crate? Can I use AI to contribute?
* What does "obviously you're not a golfer mean"??
