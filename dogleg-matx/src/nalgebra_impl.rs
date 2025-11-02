use crate::{
    Addx, Colx, Dotx, Matx, Ownedx, Scalex, Svdx, ToSvdx, TrMatVecMulx, TransformedVecNorm,
};
use nalgebra::allocator::Allocator;
use nalgebra::constraint::{AreMultipliable, DimEq, ShapeConstraint};
use nalgebra::{
    ClosedAddAssign, ClosedMulAssign, Const, DefaultAllocator, DimMin, DimSub, Matrix, OMatrix,
    OVector, Storage, UninitVector, Vector, SVD, U1,
};
use nalgebra::{Dim, Scalar};
use nalgebra::{RawStorageMut, RealField};
use num_traits::{ConstOne, Float, One, Zero};

impl<T, C, R, S> Matx<T> for Matrix<T, R, C, S>
where
    T: Scalar,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<R, C>,
    S: Storage<T, R, C>,
{
    type Owned = OMatrix<T, R, C>;

    fn into_owned(self) -> Self::Owned {
        Matrix::<_, _, _, _>::into_owned(self)
    }

    fn clone_owned(&self) -> Self::Owned {
        Matrix::<_, _, _, _>::clone_owned(self)
    }
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
    T: Scalar + RealField + Float,
    R: Dim,
    S: Storage<T, R>,
    DefaultAllocator: Allocator<R>,
{
    type Owned = OVector<T, R>;
    type Dim = usize;

    fn enorm(&self) -> T {
        crate::utility::enorm(self.iter().copied())
    }

    fn clone_owned(&self) -> Self::Owned {
        self.clone_owned()
    }

    fn into_owned(self) -> OVector<T, R> {
        Vector::<_, _, _>::into_owned(self)
    }

    fn max(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        Some(self.max())
    }

    fn dim(&self) -> Self::Dim {
        self.nrows()
    }
}

impl<T, R, S> Scalex<T> for Vector<T, R, S>
where
    T: Scalar + RealField,
    R: Dim,
    S: RawStorageMut<T, R>,
    DefaultAllocator: Allocator<R>,
{
    fn scale(mut self, factor: T) -> Self {
        self.scale_mut(factor);
        self
    }

    fn scale_mut(&mut self, factor: T) {
        self.scale_mut(factor);
    }
}

impl<T, R, C, S, D, SV> TrMatVecMulx<T, Vector<T, D, SV>> for Matrix<T, R, C, S>
where
    T: Scalar + RealField + Float + ClosedAddAssign + ClosedMulAssign + Zero + One,
    R: Dim,
    C: Dim,
    D: Dim,
    DefaultAllocator: Allocator<R, C>,
    DefaultAllocator: Allocator<R>,
    DefaultAllocator: Allocator<C>,
    DefaultAllocator: Allocator<D>,
    S: Storage<T, R, C>,
    SV: Storage<T, D>,
    ShapeConstraint: AreMultipliable<C, R, D, U1>,
{
    type Output = OVector<T, C>;

    fn tr_mulv(&self, v: &Vector<T, D, SV>) -> Option<Self::Output> {
        let (r, c) = self.shape_generic();
        let (d, _) = v.shape_generic();
        // we always have to add this check if one of the dimensions is
        // dynamically sized.
        if r.value() != d.value() {
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
    T: Scalar + RealField + Float + Zero + ClosedAddAssign + ClosedMulAssign,
    R: nalgebra::Dim,
    DefaultAllocator: Allocator<R>,
    S1: Storage<T, R>,
    S2: Storage<T, R>,
{
    fn dot(&self, v: &Vector<T, R, S1>) -> Option<T> {
        let (r1, _) = self.shape_generic();
        let (r2, _) = v.shape_generic();
        if r1 != r2 {
            return None;
        }

        Some(Vector::<_, _, _>::dot(self, v))
    }
}

impl<T, R1, R2, S1, S2> Addx<T, Vector<T, R2, S2>> for Vector<T, R1, S1>
where
    T: Scalar + RealField + Float + ClosedAddAssign + Copy + ConstOne,
    R1: nalgebra::Dim,
    R2: nalgebra::Dim,
    DefaultAllocator: Allocator<R1>,
    DefaultAllocator: Allocator<R2>,
    S1: Storage<T, R1> + RawStorageMut<T, R1>,
    S2: Storage<T, R2>,
    ShapeConstraint: DimEq<R1, R2>,
{
    fn scaled_add(mut self, factor: T, y: &Vector<T, R2, S2>) -> Option<Self> {
        let (r1, _) = self.shape_generic();
        let (r2, _) = y.shape_generic();
        if r1.value() != r2.value() {
            return None;
        }

        Vector::<_, _, _>::axpy(&mut self, T::ONE, y, factor);
        Some(self)
    }
}

impl<T, R, C, D, S, SV> TransformedVecNorm<T, Vector<T, D, SV>> for Matrix<T, R, C, S>
where
    T: Scalar + RealField + Float + ClosedAddAssign + ClosedMulAssign + Zero + One,
    R: Dim,
    C: Dim,
    D: Dim,
    DefaultAllocator: Allocator<R, C>,
    DefaultAllocator: Allocator<D>,
    S: Storage<T, R, C>,
    SV: Storage<T, D>,
    ShapeConstraint: AreMultipliable<R, C, D, U1>,
    DefaultAllocator: nalgebra::allocator::Allocator<R>,
{
    fn mulv_enorm(&self, v: &Vector<T, D, SV>) -> Option<T> {
        let (_, c) = self.shape_generic();
        let (d, _) = v.shape_generic();
        if c.value() != d.value() {
            return None;
        }

        Some((self * v).enorm())
    }
}

impl<T, R, C> ToSvdx<T> for OMatrix<T, R, C>
where
    R: Dim + DimMin<C>,
    T: RealField + Scalar + Float,
    C: Dim,
    DefaultAllocator: Allocator<R, C>,
    DefaultAllocator: Allocator<R>,
    DefaultAllocator: Allocator<C>,
    DefaultAllocator: Allocator<R, <R as DimMin<C>>::Output>,
    DefaultAllocator: Allocator<<R as DimMin<C>>::Output>,
    DefaultAllocator: Allocator<<R as DimMin<C>>::Output, C>,
    <R as DimMin<C>>::Output: DimSub<Const<1>>,
    DefaultAllocator: Allocator<<<R as DimMin<C>>::Output as DimSub<Const<1>>>::Output>,
{
    type Svd = SVD<T, R, C>;

    fn calc_svd(self) -> Option<Self::Svd> {
        SVD::try_new_unordered(
            self,
            true,
            true,
            // these are the parameters that SVD::new() uses in the nalgebra code
            <T as Float>::epsilon() * nalgebra::convert(5.0),
            0,
        )
    }
}

impl<T, R, C, SV> Svdx<T, Vector<T, R, SV>> for nalgebra::SVD<T, R, C>
where
    R: Dim + DimMin<C>,
    T: RealField + Scalar + Float,
    C: Dim,
    DefaultAllocator: Allocator<R, C>,
    DefaultAllocator: Allocator<R>,
    DefaultAllocator: Allocator<C>,
    DefaultAllocator: Allocator<R, <R as DimMin<C>>::Output>,
    DefaultAllocator: Allocator<<R as DimMin<C>>::Output>,
    DefaultAllocator: Allocator<<R as DimMin<C>>::Output, C>,
    <R as DimMin<C>>::Output: DimSub<Const<1>>,
    DefaultAllocator: Allocator<<<R as DimMin<C>>::Output as DimSub<Const<1>>>::Output>,
    SV: nalgebra::Storage<T, R>,
{
    type Output = OVector<T, C>;

    fn solve_lsqr(&self, v: &Vector<T, R, SV>) -> Option<Self::Output> {
        // since we expect non-singular matrices, this is okay. Otherwise
        // we could also be smarter and use a fraction of the largest eigenvalue,
        // like we do in nalgebra-lapack (for the QR decomposition).
        self.solve(v, Float::sqrt(Float::epsilon())).ok()
    }
}
