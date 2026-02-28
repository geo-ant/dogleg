use crate::dogleg::report::TerminationFailure;

/// error type for this crate
#[derive(thiserror::Error)]
#[error("Termination failed with reason {:?}", failure)]
pub struct Error<P> {
    /// The state of the least squares problem at termination.
    /// This might still contain parameters that are useful in practice
    /// even if the algorithm doesn't indicate convergence.
    pub problem: P,
    /// the reason for termination failure
    pub failure: TerminationFailure,
}

impl<P> std::fmt::Debug for Error<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Error")
            .field("termination", &self.failure)
            .finish_non_exhaustive()
    }
}

// impl<P, T> Error<P> where P: LeastSquaresProblem<T> {}
