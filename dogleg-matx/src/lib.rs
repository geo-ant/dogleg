mod faer_impl;
mod nalgebra_impl;

pub trait Ownedx {}

/// matrix abstraction
pub trait Matx<T> {
    type Owned: Ownedx;

    fn into_owned(self) -> Self::Owned;
    fn clone_owned(&self) -> Self::Owned;
}

/// column vector abstraction
pub trait Colx<T> {
    type Owned: Ownedx;
    fn enorm(&self) -> T;
    fn scale(self, factor: T) -> Self;
    fn clone_owned(&self) -> Self::Owned;
    fn into_owned(self) -> Self::Owned;
}

/// add / subtract from this vector
pub trait Addx<T, V>: Colx<T> + Sized
where
    V: Colx<T>,
{
    /// calculate self = a*self + b*y
    fn axpy(self, a: T, y: &V, b: T) -> Option<Self>;
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
pub trait TrMatVecMulx<T, V>: Matx<T>
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
pub trait TransformedVecNorm<T, V>: Matx<T>
where
    V: Colx<T>,
{
    fn mulv_enorm(&self, v: &V) -> Option<T>;
}

/// The singular value decomposition of this matrix can be calculated
pub trait ToSvdx<T>: Matx<T> {
    type Svd;

    fn calc_svd(self) -> Option<Self::Svd>;
}

pub trait SvdSolverx<T, V> {
    type Output: Colx<T>;
    fn solve(&self, v: &V) -> Option<Self::Output>;
}
