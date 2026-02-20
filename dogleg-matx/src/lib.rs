//! Linear algebra abstraction that are specific to the `dogleg` crate.
//! This crate has no affiliation with another crate named `matx` crate
//! on crates.io.
//!
//! You are free to implement the abstractions for a different matrix backend.
//! If you do, consider submitting a PR to this repository, so that
//! everyone can profit from this and use this crate with your
//! favorite matrix backend.

// if the cloudflare outage taught us one thing, it's that we want to be
// more strict about this...
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::panicking_unwrap)]

use num_traits::ConstOne;
use std::ops::AddAssign;

mod faer_impl;
/// utility module for floating point constants
pub mod magic_const;
mod nalgebra_impl;
mod utility;

/// indicates that a matrix or vector type owns its storage
pub trait Ownedx {}

/// matrix abstraction
pub trait Matx<T> {
    /// the equivalent type that owns its own storage
    type Owned: OwnedMatx<T>;
    /// consume `self` and give a copy in an owned type
    fn into_owned(self) -> Self::Owned;
    /// clone into an owned type
    fn clone_owned(&self) -> Self::Owned;
    /// number of columns in the matrix.
    /// Return None if the number of columns doesn't fit into u64, in which
    /// case you shouldn't be using this library anyways...
    fn ncols(&self) -> Option<u64>;
    /// number of columns in the matrix.
    /// Return None if the number of columns doesn't fit into u64, in which
    /// case you shouldn't be using this library anyways...
    fn nrows(&self) -> Option<u64>;
}

/// A matrix that owns its own storage
pub trait OwnedMatx<T>: Matx<T, Owned = Self> {}
impl<T, M> OwnedMatx<T> for M where M: Matx<T, Owned = Self> {}

/// A column vector that owns its own storage
pub trait OwnedColx<T>: Colx<T, Owned = Self> {}
impl<T, V> OwnedColx<T> for V where V: Colx<T, Owned = Self> {}

/// column vector abstraction
pub trait Colx<T>: PartialEq {
    /// the corresponding owned column vector type of same dimensions and type
    type Owned: OwnedColx<T>;
    /// calculate the euclidean norm of the vector
    fn enorm(&self) -> T;
    /// clone the values of `self` and return an owned instance.
    fn clone_owned(&self) -> Self::Owned;
    /// consume `self` and return an `Owned` instance with the same values
    fn into_owned(self) -> Self::Owned;
    /// the maximum element or `None` if the vector is empty
    fn max(&self) -> Option<T>;
    /// the number of elements in this vector.
    /// Return None if the number of columns doesn't fit into u64, in which
    /// case you shouldn't be using this library anyways...
    fn dim(&self) -> Option<u64>;
}

/// multiply a vector type by a constant factor
pub trait Scalex<T> {
    /// scale `self` by `factor`
    fn scale_mut(&mut self, factor: T);
    /// scale `self` by `factor` and return `self`
    fn scale(self, factor: T) -> Self;
}

/// add / subtract from a vector
pub trait Addx<T, V = Self>: Sized {
    /// calculate self + factor*y. Return `None` on dimension mismatch.
    fn scaled_add(self, factor: T, y: &V) -> Option<Self>;

    #[inline]
    /// calculate self + y
    fn add(self, y: &V) -> Option<Self>
    where
        T: ConstOne,
    {
        self.scaled_add(T::ONE, y)
    }
}

/// scalar (dot) product of two column vectors
pub trait Dotx<T, V = Self> {
    /// calculate the scalar (dot) product <self,v>. Returns `None` if there's
    /// a dimension mismatch
    fn dot(&self, v: &V) -> Option<T>;
}

/// For a matrix `A` that implements this, we can calculate the matrix-vector
/// product `A^T v` with a suitably sized vector.
pub trait TrMatVecMulx<T, V> {
    type Output: OwnedColx<T>;
    /// calculate `A^T v`. Returns `None` if there is a
    /// dimensions mismatch.
    fn tr_mulv(&self, v: &V) -> Option<Self::Output>;
}

/// For a matrix `A` that implements this, we can calculate the
/// euclidean norm ||A v|| of the matrix-vector product, for a suitably
/// sized vector. This can obviously be implemented for a matrix A, but
/// it can also be implemented for matrix decompositions. For QR decomposition
/// of A we have e.g.: ||A v|| = ||Q R v|| = ||R v||, because of Q^T Q = Id.
/// `R v` is cheaper to compute then `A v` because `R` is upper triangular.
pub trait TransformedVecNorm<T, V> {
    /// calculate the norm ||A v|| for a suitably sized vector v
    fn mulv_enorm(&self, v: &V) -> Option<T>;
}

/// For matrix `A` implementing this trait, its singular value decomposition
/// can be calculated.
pub trait ToSvdx<T> {
    type Svd;

    /// calculate the SVD (singular value decomposition). Return `None` on error.
    fn calc_svd(self) -> Option<Self::Svd>;
}

/// Abstracts over the singular value decomposition of a matrix `A`
pub trait Svdx<T, V> {
    type Output: Colx<T>;
    /// Solve
    /// ```math
    /// ||A x - b||^2 -> min for x
    /// ```
    ///
    /// wich solves the system
    ///
    /// ```math
    /// A x = b
    /// ```
    /// in a least squares sense. Return `None` on error.
    ///
    /// This is mathematically equivalent to solving the normal equations
    /// `A^T A x = A^T b`, but numerically much more stable.
    fn solve_lsqr(&self, b: &V) -> Option<Self::Output>;

    /// solve the regularized least squares problem
    ///
    /// ```math
    /// min ||A x - b||^2 + mu*||x||^2 -> min for x
    /// ```
    ///
    /// which is mathematically equivalent to, but numerically more efficient than
    /// solving the regularized normal equations
    ///
    /// ```math
    /// (A^T A + mu*Id) x = A^T b
    /// ```
    ///
    /// where the advantage is that `A^T A + mu* Id` is nonsingular for mu>0.
    /// For numerical reasons, mu might need to be adjusted.
    fn solve_lsqr_regularized(&self, b: &V, mu: T) -> Option<Self::Output>;

    fn rank(&self) -> usize;
}

/// calculate the column norms of a matrix and put them into a vector
/// (indexed the same as the column, so element i of the vector will have
/// norm of column i).
pub trait ColEnormsx<T> {
    type Output: OwnedColx<T>;
    /// the calculated column norms placed into avector
    fn column_enorms(&self) -> Self::Output;
}

/// used to indicate whether to invert the diagonal matrix for
/// multiplication.
pub enum Invert {
    Yes,
    No,
}

/// trait for right-multiplying a diagonal matrix `D` to a matrix
/// `A` to calculate `A D` or `A D^-1`.
pub trait DiagRightMulx<V>: Sized {
    /// returns the result `A D` or `A D^-1`, depending on the value
    /// of `invert`. `D` is given by the vector
    /// of its diagonal elements or `None`, if the dimensions are wrong
    fn mul_diag_right(self, diagonal: &V, invert: Invert) -> Option<Self>;
}

/// trait for left-multiplying a diagonal matrix `D` to a vector
/// `v` to calculate `D v` or `(D^-1) v`.
pub trait DiagLeftMulx<T, V>: Sized {
    /// returns the result `D v` or `(D^-1) v`, depending on the value
    /// of `invert`. `D` is given by the vector
    /// of its diagonal elements or `None`, if the dimensions are wrong
    /// or if the inversion could not be carried out.
    fn diag_mul_left(self, diagonal: &V, invert: Invert) -> Option<Self>;

    /// returns ||D v||, meaning the euclidean norm of the vector
    /// self times the diagonal. Very dogleg specific and we don't need
    /// to be able to invert the diagonal here.
    fn diag_mul_left_enorm(&self, diagonal: &V) -> Option<T>
    where
        T: AddAssign;
}

/// a calculation that is pretty specific to the trust region problem.
/// This is used in the gtol-criterion, which is described elsewhere in
/// this codebase.
///
/// What this calculation does is this:
/// We have two VECTORS (of same lenth) `self` and `v`, and a scalar `s`.
/// What we now calculate is:
///
/// ```math
///            self_i
/// max_i  ------------
///           s * v_i
/// ```
///
/// This seems a bit weird, but it's only used when `self` is the gradient
/// of the problem self = J^T r (such that `self_i` = `g_i` = `j_i^T r`),
/// and `v` = the column norms of `J`, and `s = ||r||` (the residual norm).
/// This is how this gets used for the gtol criterion, see also
/// MINPACK user guide p.22. We can assume `s!=0`, because we'll have checked
/// earlier if the residuals vanished (in which case iteration is successful).
pub trait MaxScaledDivx<T, V> {
    /// calculation as described above where None means the vectors
    /// had no elements. The implementation is free to assume that the
    /// vectors have same length.
    fn max_abs_scaled_div(&self, s: T, v: &V) -> Option<T>;
}

/// a very dogleg specific trait that is used when assigning the diagonal
/// scaling during iterations. So this needs to be only implemented for
/// vectors, which is what we use to store the diagonal scaling matrix
/// anyways.
///
/// What this does is assign every element self_i = max(self_i, other_i).
/// Then returns that modified self or None if an error occurred (dimensions
/// mismatch).
pub trait ElementwiseMaxx<V>: Sized {
    fn elementwise_max(self, other: &V) -> Option<Self>;
}

/// another very dogleg specific trait that replaces values smaller or equal
/// to eps with a replacement value.
/// This is used on initial assignment for the diagonal scaling matrix.
pub trait ElementwiseReplaceLeqx<T> {
    /// replace all elements less or equal to `threshold` with `replacement`
    /// and return self again.
    fn replace_if_leq(self, threshold: T, replacement: T) -> Self;
    /// clamp all the elements in the vector to be between `min` and `max` 
    /// and return self again.
    fn clamp(self, min: T, max: T) -> Self;
}
