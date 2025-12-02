use approx::{AbsDiffEq, RelativeEq};
use faer::{col::AsColRef, mat::AsMatRef, ColRef, MatRef, Shape};

/// utility macro because faer doesn't natively expose facilities to work
/// with the approx crate.
#[macro_export]
macro_rules! col_assert_relative_eq {
    ($lhs:expr,$rhs:expr $(, epsilon = $eps:literal)? $(,)?) => {
        ::approx::assert_relative_eq!(
            $crate::faer_impl::approx_util::approx_col(& $lhs),
            $crate::faer_impl::approx_util::approx_col(& $rhs)
            $(, epsilon = $eps)?)
    };
}

#[macro_export]
/// helper macro to use instead of approx::relative_eq! for faer column vectors
macro_rules! col_relative_eq {
    ($lhs:expr,$rhs:expr $(, epsilon = $eps:literal)?) => {
        ::approx::relative_eq!(
            $crate::faer_impl::approx_util::approx_col(& $lhs),
            $crate::faer_impl::approx_util::approx_col(& $rhs)
            $(, epsilon = $eps)?)
    };
}

#[macro_export]
macro_rules! mat_assert_relative_eq {
    ($lhs:expr,$rhs:expr $(, epsilon = $eps:literal)? $(,)?) => {
        ::approx::assert_relative_eq!(
            $crate::faer_impl::approx_util::approx_mat(& $lhs),
            $crate::faer_impl::approx_util::approx_mat(& $rhs)
            $(, epsilon = $eps)?)
    };
}
/// utility structure that implements the traits from the approx crate for
/// faer column vectors such that I can use them in the respective macros
/// of the approx crate.
///
/// They are internally implemented in faer, but cannot be exposed to the outside
/// and according to the lead dev, this is about to change in future anyways.
#[derive(Clone, PartialEq)]
pub struct ApproxColWrapper<'a, T, R>(ColRef<'a, T, R>)
where
    R: Shape;

impl<'a, T: std::fmt::Debug, R: Shape> std::fmt::Debug for ApproxColWrapper<'a, T, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// same as the col ref wrapper
#[derive(Clone, PartialEq)]
pub struct ApproxMatWrapper<'a, T, R, C>(MatRef<'a, T, R, C>)
where
    C: Shape,
    R: Shape;

impl<'a, T: std::fmt::Debug, R: Shape, C: Shape> std::fmt::Debug for ApproxMatWrapper<'a, T, R, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

pub fn approx_col<T, R, V>(val: &V) -> ApproxColWrapper<'_, T, R>
where
    V: AsColRef<T = T, Rows = R>,
    R: Shape,
{
    ApproxColWrapper(val.as_col_ref())
}

pub fn approx_mat<T, R, C, M>(mat: &M) -> ApproxMatWrapper<'_, T, R, C>
where
    M: AsMatRef<T = T, Rows = R, Cols = C>,
    R: Shape,
    C: Shape,
{
    ApproxMatWrapper(mat.as_mat_ref())
}

impl<'a, T, R: Shape> AbsDiffEq for ApproxColWrapper<'a, T, R>
where
    T: AbsDiffEq,
    T::Epsilon: Copy,
{
    type Epsilon = <T as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        if self.0.nrows() != other.0.nrows() {
            return false;
        }
        self.0
            .iter()
            .zip(other.0.iter())
            .all(move |(lhs, rhs)| lhs.abs_diff_eq(rhs, epsilon))
    }
}

impl<'a, T, R> RelativeEq for ApproxColWrapper<'a, T, R>
where
    T: RelativeEq,
    T::Epsilon: Copy,
    R: Shape,
{
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .all(move |(lhs, rhs)| lhs.relative_eq(rhs, epsilon, max_relative))
    }
}

impl<'a, T, R: Shape, C: Shape> AbsDiffEq for ApproxMatWrapper<'a, T, R, C>
where
    T: AbsDiffEq,
    T::Epsilon: Copy,
{
    type Epsilon = <T as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        if self.0.nrows() != other.0.nrows() {
            return false;
        }
        self.0
            .col_iter()
            .zip(other.0.col_iter())
            .all(move |(lhs, rhs)| {
                lhs.iter()
                    .zip(rhs.iter())
                    .all(|(lhs, rhs)| lhs.abs_diff_eq(rhs, epsilon))
            })
    }
}

impl<'a, T, R, C> RelativeEq for ApproxMatWrapper<'a, T, R, C>
where
    T: RelativeEq,
    T::Epsilon: Copy,
    R: Shape,
    C: Shape,
{
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.0
            .col_iter()
            .zip(other.0.col_iter())
            .all(move |(lhs, rhs)| {
                lhs.iter()
                    .zip(rhs.iter())
                    .all(|(lhs, rhs)| lhs.relative_eq(rhs, epsilon, max_relative))
            })
    }
}
