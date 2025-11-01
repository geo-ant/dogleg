//! Linear algebra abstraction that are specific to the `dogleg` crate.
//! This crate has no affiliation with another crate named `matx` crate
//! on crates.io.
//!
//! You are free to implement the abstractions for a different matrix backend.
//! If you do, consider submitting a PR to this repository, so that
//! everyone can profit from this and use this crate with your
//! favorite matrix backend.

mod faer_impl;
mod nalgebra_impl;
mod utility;

/// indicates that a matrix or vector type owns its storage
pub trait Ownedx {}

/// matrix abstraction
pub trait Matx<T> {
    /// the equivalent type that owns its own storage
    type Owned: Ownedx;
    /// consume `self` and give a copy in an owned type
    fn into_owned(self) -> Self::Owned;
    /// clone into an owned type
    fn clone_owned(&self) -> Self::Owned;
}

/// calculate the column norms of a matrix and put them into a vector
/// (indexed the same as the column, so element i of the vector will have
/// norm of column i).
pub trait ColEnormsx<T>: Matx<T> {
    type Norms: OwnedColx<T>;
    /// the calculated column norms placed into avector
    fn column_enorms(&self) -> Option<Self::Norms>;
}

/// used to indicate whether to invert the diagonal matrix for
/// multiplication.
pub enum Invert {
    Yes,
    No,
}

/// trait for right-multiplying a diagonal matrix `D` to a matrix
/// `A` to calculate `A D` or `A D^-1`.
pub trait DiagRightMulx<T, V>: Sized
where
    V: Colx<T>,
{
    /// returns the result `A D` or `A D^-1`, depending on the value
    /// of `invert`. `D` is given by the vector
    /// of its diagonal elements or `None`, if the dimensions are wrong
    fn diag_right_mul(self, diagonal: &V, invert: Invert) -> Option<Self>;
}

/// trait for left-multiplying a diagonal matrix `D` to a vector
/// `v` to calculate `D v` or `(D^-1) v`.
pub trait DiagLeftMulx<T, V>: Sized {
    /// returns the result `D v` or `(D^-1) v`, depending on the value
    /// of `invert`. `D` is given by the vector
    /// of its diagonal elements or `None`, if the dimensions are wrong
    /// or if the inversion could not be carried out.
    fn diag_left_mul(self, diagonal: &V, invert: Invert) -> Option<Self>;
}

/// component wise multiplication. Doesn't need to be implemented
/// manually since it is blanket implemented through the DiagLeftMultiply
/// trait, which will be implemented for vectors anyways. So this is
/// just an alias to make the code more readable.
pub trait ComponentMulx<T, V>: Sized {
    /// multiply `self` and V component wise. Return `None` on dimension
    /// mismatch.
    fn component_mul(self, v: &V) -> Option<Self>;
    /// divide self and V component wise
    fn component_div(self, v: &V) -> Option<Self>;
}

/// Blanket impl
impl<T, V1, V2> ComponentMulx<T, V2> for V1
where
    V1: Sized + DiagLeftMulx<T, V2>,
{
    #[inline]
    fn component_mul(self, v: &V2) -> Option<Self> {
        self.diag_left_mul(v, Invert::No)
    }
    #[inline]
    fn component_div(self, v: &V2) -> Option<Self> {
        self.diag_left_mul(v, Invert::Yes)
    }
}

/// A matrix that owns its own storage
pub trait OwnedMatx<T>: Matx<T> {}
impl<T, M> OwnedMatx<T> for M where M: Matx<T, Owned = Self> {}

/// A column vector that owns its own storage
pub trait OwnedColx<T>: Colx<T> {}
impl<T, V> OwnedColx<T> for V where V: Colx<T, Owned = Self> {}

/// column vector abstraction
pub trait Colx<T> {
    /// the corresponding owned column vector type of same dimensions and type
    type Owned: Ownedx;
    /// an unsigned integer type for the number of elements in the vector.
    /// Often e.g. usize.
    type Dim;
    /// calculate the euclidean norm of the vector
    fn enorm(&self) -> T;
    /// clone the values of `self` and return an owned instance.
    fn clone_owned(&self) -> Self::Owned;
    /// consume `self` and return an `Owned` instance with the same values
    fn into_owned(self) -> Self::Owned;
    /// the maximum element or `None` if the vector is empty
    fn max(&self) -> Option<T>;
    /// the number of elements in this vector
    fn dim(&self) -> Self::Dim;
}

/// multiply a matrix or vector type by a constant factor
pub trait Scalex<T> {
    /// scale `self` by `factor`
    fn scale_mut(&mut self, factor: T);
    /// scale `self` by `factor` and return `self`
    fn scale(self, factor: T) -> Self;
}

/// add / subtract from this vector
pub trait Addx<T, V = Self>: Sized
where
    V: Colx<T>,
{
    /// calculate self + factor*y. Return `None` on dimension mismatch.
    fn scaled_add(self, factor: T, y: &V) -> Option<Self>;
}

/// scalar (dot) product of two column vectors
pub trait Dotx<T, V = Self>: Colx<T>
where
    V: Colx<T>,
{
    /// calculate the scalar (dot) product <self,v>. Returns `None` if there's
    /// a dimension mismatch
    fn dot(&self, v: &V) -> Option<T>;
}

/// For a matrix `A` that implements this, we can calculate the matrix-vector
/// product `A^T v` with a suitably sized vector.
pub trait TrMatVecMulx<T, V>
where
    V: Colx<T>,
{
    type Output: Colx<T>;
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
pub trait TransformedVecNorm<T, V>
where
    V: Colx<T>,
{
    /// calculate the norm ||A v|| for a suitably sized vector v
    fn mulv_enorm(&self, v: &V) -> Option<T>;
}

/// The singular value decomposition of this matrix can be calculated
pub trait ToSvdx<T> {
    type Svd;

    /// calculate the SVD (singular value decomposition). Return `None` on error.
    fn calc_svd(self) -> Option<Self::Svd>;
}

/// Abstracts over the singular value decomposition of a matrix `A`
pub trait Svdx<T, V> {
    type Output: Colx<T>;
    /// Solve ||A x - v||^2 -> min for x. This solves the system
    /// A x = v in a least squares sense. Return `None` on error.
    fn solve_lsqr(&self, v: &V) -> Option<Self::Output>;
}
