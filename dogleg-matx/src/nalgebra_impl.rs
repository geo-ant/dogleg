use crate::utility::enorm;
use crate::{
    Addx, ColEnormsx, Colx, DiagLeftMulx, DiagRightMulx, Dotx, ElementwiseMaxx,
    ElementwiseReplaceLeqx, Invert, Matx, MaxAbsx, Ownedx, Scalex, Svdx, ToSvdx, TrMatVecMulx,
    TransformedVecNorm,
};
#[cfg(feature = "assert2")]
use assert2::debug_assert;

use nalgebra::allocator::Allocator;
use nalgebra::constraint::{AreMultipliable, DimEq, ShapeConstraint};
use nalgebra::{
    ClosedAddAssign, ClosedMulAssign, Const, DefaultAllocator, Dim, DimMin, DimSub, Matrix,
    OMatrix, OVector, Storage, UninitVector, Vector, SVD, U1,
};
use nalgebra::{RawStorage, Scalar};
use nalgebra::{RawStorageMut, RealField};
use num_traits::float::TotalOrder;
use num_traits::{ConstOne, Float, One, Zero};
use std::cmp::Ordering;
use std::ops::{AddAssign, Div, Mul};

#[cfg(test)]
mod test;

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

    fn ncols(&self) -> Option<u64> {
        self.ncols().try_into().ok()
    }

    fn nrows(&self) -> Option<u64> {
        self.nrows().try_into().ok()
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

    fn dim(&self) -> Option<u64> {
        self.nrows().try_into().ok()
    }

    fn max_absolute(&self) -> Option<T>
    where
        T: TotalOrder,
    {
        self.iter()
            .copied()
            .map(Float::abs)
            .max_by(TotalOrder::total_cmp)
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
    DefaultAllocator: Allocator<C>,
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

impl<T, R1, R2, S1, S2> Dotx<T, Vector<T, R1, S1>> for Vector<T, R2, S2>
where
    T: Scalar + RealField + Float + Zero + ClosedAddAssign + ClosedMulAssign,
    R1: Dim,
    R2: Dim,
    S1: Storage<T, R1>,
    S2: Storage<T, R2>,
    ShapeConstraint: DimEq<R1, R2>,
{
    fn dot(&self, v: &Vector<T, R1, S1>) -> Option<T> {
        let (r1, _) = self.shape_generic();
        let (r2, _) = v.shape_generic();
        if r1.value() != r2.value() {
            return None;
        }

        Some(Vector::<_, _, _>::dot(v, self))
    }
}

impl<T, R1, R2, S1, S2> Addx<T, Vector<T, R2, S2>> for Vector<T, R1, S1>
where
    T: Scalar + RealField + Float + ClosedAddAssign + Copy + ConstOne,
    R1: Dim,
    R2: Dim,
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

        Vector::<_, _, _>::axpy(&mut self, factor, y, T::ONE);
        Some(self)
    }
}

impl<T, R, C, D, S, SV> TransformedVecNorm<T, Vector<T, D, SV>> for Matrix<T, R, C, S>
where
    T: Scalar + RealField + Float + ClosedAddAssign + ClosedMulAssign + Zero + One,
    R: Dim,
    C: Dim,
    D: Dim,
    S: Storage<T, R, C>,
    SV: Storage<T, D>,
    ShapeConstraint: AreMultipliable<R, C, D, U1>,
    DefaultAllocator: Allocator<R>,
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
        let svd = SVD::try_new_unordered(
            self,
            true,
            true,
            // these are the parameters that SVD::new() uses in the nalgebra code
            <T as Float>::epsilon() * nalgebra::convert(5.0),
            0,
        )?;
        Some(svd)
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
    DefaultAllocator: Allocator<<<R as DimMin<C>>::Output as DimSub<Const<1>>>::Output>,
    <R as DimMin<C>>::Output: DimSub<Const<1>>,
    SV: Storage<T, R>,
{
    type Output = OVector<T, C>;

    fn solve_lsqr(&self, b: &Vector<T, R, SV>) -> Option<Self::Output> {
        // since we expect non-singular matrices, this is okay. Otherwise
        // we could also be smarter and use a fraction of the largest eigenvalue,
        // like we do in nalgebra-lapack (for the QR decomposition).
        self.solve(b, Float::sqrt(Float::epsilon())).ok()
    }

    fn solve_lsqr_regularized(&self, b: &Vector<T, R, SV>, mu: T) -> Option<Self::Output> {
        let v_t = self.v_t.as_ref()?;
        let u = self.u.as_ref()?;

        debug_assert!(
            mu.is_positive() && mu.is_finite(),
            "regularization parameter must be positive"
        );

        // NOTE(geo-ant) the theory
        //
        // we want to solve the regularized normal equations
        // (A^T A + mu I) x = A^T b
        //
        // The compact SVD is A = U S V^T
        //
        // where U^T U = I
        // and V^T V = V V^T = I
        // (but U U^T is not unitary)
        //
        // Plugging the SVD into the normal equations gives
        // (after some manipulation)
        //
        // => (V S^2 V^T + mu I) x = V S U^T b  | multiply V^T from the right and do some more manipulation
        // => (S^2 + mu I) V^T x = S U^T b
        //
        // setting y = V^T x  (unknown)
        // and z = U^T b (known)
        //
        // (S^2 + mu I) y = S z
        //
        // this is a diagonal equation for y and z, so that we can set
        //
        // y_j = sigma_j / (sigma_j^2 + mu) * z_j
        //
        // which gives us y. And to get x from y, we calculate
        //
        // x = V y

        // the vector of sigma_j / (sigma_j^2 + mu)
        let smu = self
            .singular_values
            .map(|sigma| sigma / (Float::powi(sigma, 2) + mu));

        let z = u.tr_mul(b);
        let y = z.component_mul(&smu);
        let x = v_t.tr_mul(&y);
        Some(x)
    }
}

impl<T, R, C, S> ColEnormsx<T> for Matrix<T, R, C, S>
where
    T: Scalar + RealField + Float + Copy + ConstOne,
    C: Dim,
    R: Dim,
    DefaultAllocator: Allocator<C>,
    S: Storage<T, R, C>,
{
    type Output = OVector<T, C>;

    fn column_enorms(&self) -> Self::Output {
        let (_, c) = self.shape_generic();
        OVector::<T, C>::from_iterator_generic(
            c,
            U1,
            self.column_iter()
                .map(|col| crate::utility::enorm(col.iter().copied())),
        )
    }

    fn damped_inverse_column_enorms(&self) -> Self::Output {
        let (_, c) = self.shape_generic();
        OVector::<T, C>::from_iterator_generic(
            c,
            U1,
            self.column_iter().map(|col| {
                let col_norm = crate::utility::enorm(col.iter().copied());
                T::ONE / (T::ONE + col_norm)
            }),
        )
    }
}

impl<T, RM, CM, SM, RV, SV> DiagRightMulx<Vector<T, RV, SV>> for Matrix<T, RM, CM, SM>
where
    T: RealField + Scalar + Float + Copy,
    RM: Dim,
    CM: Dim,
    RV: Dim,
    SV: Storage<T, RV>,
    SM: Storage<T, RM, CM> + RawStorageMut<T, RM, CM>,
    ShapeConstraint: AreMultipliable<RM, CM, RV, U1>,
{
    fn mul_diag_right(mut self, diagonal: &Vector<T, RV, SV>, invert: Invert) -> Option<Self> {
        let (_, c) = self.shape_generic();
        let (d, _) = diagonal.shape_generic();
        if c.value() != d.value() {
            return None;
        }

        // right multiplying matrix `self` with a diagonal matrix means scaling
        // the columns.
        self.column_iter_mut()
            .zip(diagonal.iter().copied())
            .for_each(|(mut col, diag)| {
                let diag = match invert {
                    Invert::Yes => Float::powi(diag, -1),
                    Invert::No => diag,
                };
                col *= diag;
            });

        Some(self)
    }
}

impl<T, R2, S2, R1, S1> DiagLeftMulx<T, Vector<T, R1, S1>> for Vector<T, R2, S2>
where
    T: RealField + Scalar + Float + Copy,
    R2: Dim,
    R1: Dim,
    S1: Storage<T, R1>,
    S2: Storage<T, R2> + RawStorageMut<T, R2>,
    ShapeConstraint: DimEq<R1, R2>,
{
    fn diag_mul_left(mut self, diagonal: &Vector<T, R1, S1>, invert: Invert) -> Option<Self> {
        let (r2, _) = self.shape_generic();
        let (r1, _) = diagonal.shape_generic();
        if r1.value() != r2.value() {
            return None;
        }

        self.iter_mut()
            .zip(diagonal.iter().copied())
            .for_each(|(elem, diag)| {
                let diag = match invert {
                    Invert::Yes => Float::powi(diag, -1),
                    Invert::No => diag,
                };
                *elem *= diag;
            });

        Some(self)
    }

    fn diag_mul_left_enorm(&self, diagonal: &Vector<T, R1, S1>) -> Option<T>
    where
        T: AddAssign,
    {
        let (r2, _) = self.shape_generic();
        let (r1, _) = diagonal.shape_generic();
        if r1.value() != r2.value() {
            return None;
        }

        Some(enorm(
            self.iter()
                .copied()
                .zip(diagonal.iter().copied())
                .map(|(elem, diag)| elem * diag),
        ))
    }
}

impl<T, R1, R2, S1, S2> MaxAbsx<T, Vector<T, R2, S2>> for Vector<T, R1, S1>
where
    T: Float + Scalar + Copy + Mul<Output = T> + Div<Output = T> + TotalOrder,
    R1: Dim,
    R2: Dim,
    S1: Storage<T, R1> + RawStorage<T, R1>,
    S2: Storage<T, R2>,
{
    fn max_abs_scaled_div_elem(&self, s: T, v: &Vector<T, R2, S2>) -> Option<T> {
        self.iter()
            .copied()
            .zip(v.iter().copied())
            .map(|(this_i, vi)| this_i.abs() / vi)
            .max_by(TotalOrder::total_cmp)
            .map(|val| val / s)
    }
}

impl<T, R1, R2, S1, S2> ElementwiseMaxx<Vector<T, R2, S2>> for Vector<T, R1, S1>
where
    T: Scalar + Copy + TotalOrder,
    R1: Dim,
    R2: Dim,
    ShapeConstraint: DimEq<R1, R2>,
    S1: Storage<T, R1> + RawStorageMut<T, R1>,
    S2: Storage<T, R2>,
{
    fn elementwise_max(mut self, other: &Vector<T, R2, S2>) -> Option<Self> {
        let (r1, _) = self.shape_generic();
        let (r2, _) = other.shape_generic();
        if r1.value() != r2.value() {
            return None;
        }

        self.iter_mut().zip(other.iter()).for_each(|(this, elem)| {
            let max = match TotalOrder::total_cmp(this, elem) {
                std::cmp::Ordering::Less => *elem,
                _ => *this,
            };
            *this = max;
        });
        Some(self)
    }
}

impl<T, R, S> ElementwiseReplaceLeqx<T> for Vector<T, R, S>
where
    T: Float + Scalar + Copy + TotalOrder,
    R: Dim,
    S: Storage<T, R> + RawStorageMut<T, R>,
{
    fn replace_if_leq(mut self, threshold: T, replacement: T) -> Self {
        self.iter_mut().for_each(|elem| {
            if !matches!(elem.total_cmp(&threshold), Ordering::Greater) {
                *elem = replacement;
            }
        });
        self
    }

    fn clamp(mut self, min: T, max: T) -> Self {
        self.iter_mut().for_each(|elem| {
            *elem = Float::clamp(*elem, min, max);
        });
        self
    }
}
