use crate::LeastSquaresProblem;

/// helper type that provides an RAII guard type that resets the parameters of
/// the given problem to the original parameters set on construction, unless it
/// was explicitly defused. This is only
pub struct ProblemResetGuard<'a, T, P>
where
    P: LeastSquaresProblem<T>,
{
    problem: &'a mut P,
    params: Option<P::Parameters>,
}

/// update the parameters of a least squares problems and remember the old parameters.
/// If the guard isn't defused, the
#[must_use = "keep the guard in a named variable, otherwise the parameters will be immediately reset"]
pub fn update_params<T, P: LeastSquaresProblem<T>>(
    problem: &mut P,
    new_params: P::Parameters,
) -> ProblemResetGuard<'_, T, P> {
    let guard = ProblemResetGuard::new(problem);
    guard.problem.set_params(new_params);
    guard
}

impl<'a, T, P> ProblemResetGuard<'a, T, P>
where
    P: LeastSquaresProblem<T>,
{
    /// construct a new instance and remember the currently set parameters. They
    /// will be set again on drop, unless this guard was defused
    fn new(problem: &'a mut P) -> Self {
        let params = Some(problem.params());
        Self { problem, params }
    }

    /// if this is called, then the
    pub fn defuse(&mut self) {
        debug_assert!(
            self.params.is_some(),
            "defuse called twice, indicates a logic error!"
        );
        _ = self.params.take()
    }

    /// return the currently set parameters (which are the new, updated parameters)
    pub fn params(&self) -> P::Parameters {
        self.problem.params()
    }

    pub fn residuals(&self) -> Option<P::Residuals> {
        self.problem.residuals()
    }
}

impl<'a, T, P> Drop for ProblemResetGuard<'a, T, P>
where
    P: LeastSquaresProblem<T>,
{
    fn drop(&mut self) {
        if let Some(params) = self.params.take() {
            self.problem.set_params(params);
        }
    }
}
