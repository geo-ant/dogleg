use crate::dogleg::report::TerminationFailure;

/// error type for this crate
#[derive(thiserror::Error)]
#[error("foo")]
pub struct Error<P> {
    pub problem: P, // pub termination_error:
    pub termination: TerminationFailure,
}

impl<P> std::fmt::Debug for Error<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Error")
            .field("termination", &self.termination)
            .finish_non_exhaustive()
    }
}
