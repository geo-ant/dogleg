use crate::{Addx, Colx, Dotx, Matx, Ownedx, TrMatVecMulx};
use nalgebra::allocator::Allocator;
use nalgebra::constraint::{AreMultipliable, DimEq, SameNumberOfRows, ShapeConstraint};
use nalgebra::{
    ClosedAddAssign, ClosedMulAssign, Const, DMatrix, DVector, DefaultAllocator, Matrix, OMatrix,
    OVector, RawStorage, Storage, U1, UninitVector, Vector,
};
use nalgebra::{Dim, Scalar};
use nalgebra::{RawStorageMut, RealField};
use num_traits::{One, Zero};
use std::process::Output;

impl<T, C, R, S> Matx<T> for Matrix<T, C, R, S>
where
    T: Scalar,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<R, C>,
{
    type Owned = OMatrix<T, R, C>;
}

impl<T, C, R, S> Matx<T> for &Matrix<T, C, R, S>
where
    T: Scalar,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<R, C>,
{
    type Owned = OMatrix<T, R, C>;
}

impl<T, R, C> Ownedx for OMatrix<T, R, C>
where
    T: Scalar,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<R, C>,
{
}

impl<T, R, S> Colx<T> for Vector<T, R, S>
where
    T: Scalar + RealField,
    R: Dim,
    DefaultAllocator: Allocator<R>,
    S: Storage<T, R> + RawStorageMut<T, R> + RawStorage<T, R>,
    DefaultAllocator: Allocator<R>,
{
    type Ownedx = OVector<T, R>;

    fn enormx(&self) -> T {
        todo!()
    }

    fn scalex(mut self, factor: T) -> Self {
        self.scale_mut(factor);
        self
    }

    fn clone_ownedx(&self) -> Self::Ownedx {
        self.clone_owned()
    }

    fn into_ownedx(self) -> OVector<T, R> {
        self.into_owned()
    }
}

impl<T, R, C, S, SV> TrMatVecMulx<T, Vector<T, R, SV>> for &Matrix<T, R, C, S>
where
    T: Scalar + RealField + ClosedAddAssign + ClosedMulAssign + Zero + One,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<R, C>,
    DefaultAllocator: Allocator<C, R>,
    DefaultAllocator: Allocator<R>,
    S: RawStorage<T, R, C>,
    DefaultAllocator: Allocator<C>,
    S: Storage<T, R, C>,
    SV: Storage<T, R>,
    ShapeConstraint: AreMultipliable<C, R, R, U1>,
{
    type Output = OVector<T, C>;

    fn tr_mulv(self, v: &Vector<T, R, SV>) -> Option<Self::Output> {
        let (r, c) = self.shape_generic();
        let (d, _) = v.shape_generic();
        // we always have to add this check if one of the dimensions is
        // dynamically sized.
        if r != d {
            return None;
        }

        // SAFETY safe because this is never read since we assign zero to the parameter beta in gemv
        let mut result = unsafe { UninitVector::<T, C>::uninit(c, U1).assume_init() };
        result.gemm_tr(T::one(), self, v, T::zero());

        Some(result)
    }
}

impl<T, R, S1, S2> Dotx<T, Vector<T, R, S1>> for Vector<T, R, S2>
where
    T: Scalar + RealField + Zero + ClosedAddAssign + ClosedMulAssign,
    R: nalgebra::Dim,
    DefaultAllocator: Allocator<R>,
    S1: Storage<T, R> + RawStorage<T, R> + RawStorageMut<T, R>,
    S2: Storage<T, R> + RawStorage<T, R> + RawStorageMut<T, R>,
{
    fn dot(&self, v: &Vector<T, R, S1>) -> T {
        Vector::<_, _, _>::dot(self, v)
    }
}

impl<T, R, S1, S2> Addx<T, Vector<T, R, S2>> for Vector<T, R, S1>
where
    T: Scalar + RealField + ClosedAddAssign + Copy,
    R: nalgebra::Dim,
    DefaultAllocator: Allocator<R>,
    S1: Storage<T, R> + RawStorage<T, R> + RawStorageMut<T, R>,
    S2: Storage<T, R> + RawStorage<T, R> + RawStorageMut<T, R>,
{
    fn axpy(mut self, a: T, y: &Vector<T, R, S2>, b: T) -> Option<Self> {
        let (r1, _) = self.shape_generic();
        let (r2, _) = y.shape_generic();
        if r1 != r2 {
            return None;
        }

        Vector::<_, _, _>::axpy(&mut self, a, y, b);
        Some(self)
    }
}

fn test_tr_mat_vec_mulx<T, M, V>(mat: M, v: V)
where
    M: Matx<T> + TrMatVecMulx<T, V>,
{
    mat.tr_mulv(&v);
}

fn some_tests() {
    let mat = DMatrix::<f64>::zeros(3, 4);
    let v = DVector::<f64>::from_element(4, 1.0);
    test_tr_mat_vec_mulx(&mat, v);
}
