# Test Problems for Unconstrained Minimization

This collects the sources that I found for the various test problems in
this test suite. The test suite itself is based on the test suite of the
`levenberg-marquardt` crate.

## Resources

* **Classic Paper**: The paper _Testing Unconstrained Optimization Software_
  by JJ More _et al_ contains the problems used in this suite. See e.g.
  [here](https://www.cmor-faculty.rice.edu/~yzhang/caam454/nls/MGH.pdf).
  This is often called the `MGH` benchmark.
* The [`funconstrain`](https://rdrr.io/github/jlmelville/funconstrain/)
  R package contains implementations of these problems as well as extra notes
  that can come in handy from time to time.
* The NIST also has a list of problems with certified solutions
  [here](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml).
  I might use them later as well, but for now I'm going to stick to the problems
  that the `levenberg-marquardt` crate used as well. The problems are
  specifically nonlinear least squares _regression_ problems, but those can
  easily be casted to nonlinear minimization problems, of course.
* Some other in-depth notes on the MGH benchmark
  [here](https://scilab.gitlab.io/legacy_wiki/PerformancesScilabMoreGarbowHillstromBenchmark.html).
* Just found a Rust crate [`mgh`](https://crates.io/crates/mgh). It's possible
  to view the source, but there's not repository linked. Glancing over it,
  this doesn't seem to be very high quality, but might be helpful for reference
  at some point... **don't use, this is amateur work**. A little
  detective work reveals the repo [on github here](https://github.com/fepfitra/mgh).

> **ATTENTION**: objective function / cost function.
> The sources above minimize the _sum of squares_, whereas my objective function
> is `0.5 * (sum of squares)`. Just keep this scale factor in mind.

