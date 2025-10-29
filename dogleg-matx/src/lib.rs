mod faer_impl;
mod nalgebra_impl;

/// indicates that a matrix or vector type owns its storage
pub trait Ownedx {}

/// matrix abstraction
pub trait Matx<T> {
    type Owned: Ownedx;

    /// consume `self` and give a copy in an owned type
    fn into_owned(self) -> Self::Owned;
    /// clone into an owned type
    fn clone_owned(&self) -> Self::Owned;
}

/// column vector abstraction
pub trait Colx<T> {
    type Owned: Ownedx;
    fn enorm(&self) -> T;
    fn clone_owned(&self) -> Self::Owned;
    fn into_owned(self) -> Self::Owned;
}

/// multiply a matrix or vector type by a constant factor
pub trait Scalex<T> {
    fn scale(self, factor: T) -> Self;
}

/// add / subtract from this vector
pub trait Addx<T, V>: Sized
where
    V: Colx<T>,
{
    /// calculate self + factor*y
    fn scaled_add(self, factor: T, y: &V) -> Option<Self>;
}

/// scalar (dot) product of two column vectors
pub trait Dotx<T, V>: Colx<T>
where
    V: Colx<T>,
{
    /// calculate the scalar (dot) product <self,v>
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
    fn mulv_enorm(&self, v: &V) -> Option<T>;
}

/// The singular value decomposition of this matrix can be calculated
pub trait ToSvdx<T> {
    type Svd;

    fn calc_svd(self) -> Option<Self::Svd>;
}

pub trait SvdSolverx<T, V> {
    type Output: Colx<T>;
    fn solve(&self, v: &V) -> Option<Self::Output>;
}
