mod faer_impl;
mod nalgebra_impl;

pub trait Ownedx {}

/// matrix abstraction
pub trait Matx<T> {
    type Owned: Ownedx;
}

/// column vector abstraction
pub trait Colx<T> {
    type Owned: Ownedx;
    fn enormx(&self) -> T;
    fn scalex(self, factor: T) -> Self;
    fn clone_ownedx(&self) -> Self::Owned;
    fn into_ownedx(self) -> Self::Owned;
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
    fn dot(&self, v: &V) -> T;
}

/// A matrix `A` that implements this can calculate the matrix-vector
/// product `A^T v` with a suitably sized vector.
pub trait TrMatVecMulx<T, V>: Matx<T> {
    type Output: Colx<T>;
    /// calculate `A^T v`. Returns `None` if there is a
    /// dimensions mismatch.
    fn tr_mulv(self, v: &V) -> Option<Self::Output>;
}
