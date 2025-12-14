use crate::utility::enorm;
use crate::{
    Addx, ColEnormsx, Colx, DiagLeftMulx, DiagRightMulx, Dotx, ElementwiseMaxx,
    ElementwiseReplaceLeqx, Invert, Matx, MaxScaledDivx, Ownedx, Scalex, Svdx, ToSvdx,
    TrMatVecMulx, TransformedVecNorm,
};
use faer::col::AsColMut;
use faer::linalg::solvers::Svd;
use faer::mat::{AsMatMut, AsMatRef};
use faer::prelude::SolveLstsq;
use faer::{col::AsColRef, traits::RealField, Col, ColMut, ColRef, Mat, Scale};
use faer::{Accum, MatMut, MatRef, Shape};
use num_traits::float::TotalOrder;
use num_traits::{ConstOne, Float};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::ops::{AddAssign, MulAssign};

#[cfg(test)]
mod test;

#[cfg(test)]
pub mod approx_util;

/// marker trait to get around some conflicting impl trouble with nalgebra
/// implementations.
pub trait FaerType {}
impl<T, R, C> FaerType for Mat<T, R, C>
where
    T: Clone,
    R: faer::Shape,
    C: faer::Shape,
{
}

impl<T, R, C> FaerType for MatRef<'_, T, R, C>
where
    T: Clone,
    R: faer::Shape,
    C: faer::Shape,
{
}

impl<T, R, C> FaerType for MatMut<'_, T, R, C>
where
    T: Clone,
    R: faer::Shape,
    C: faer::Shape,
{
}

impl<T, R> FaerType for Col<T, R>
where
    T: Clone,
    R: faer::Shape,
{
}

impl<T, R> FaerType for ColRef<'_, T, R>
where
    T: Clone,
    R: faer::Shape,
{
}

impl<T, R> FaerType for ColMut<'_, T, R>
where
    T: Clone,
    R: faer::Shape,
{
}

impl<T, R, C> Matx<T> for Mat<T, R, C>
where
    T: Clone,
    R: faer::Shape,
    C: faer::Shape,
    u64: TryFrom<C> + TryFrom<R>,
{
    type Owned = Self;

    fn into_owned(self) -> Self::Owned {
        self
    }

    fn clone_owned(&self) -> Self::Owned {
        self.clone()
    }

    fn ncols(&self) -> Option<u64> {
        self.ncols().try_into().ok()
    }

    fn nrows(&self) -> Option<u64> {
        self.nrows().try_into().ok()
    }
}

impl<T, R, C> Matx<T> for MatRef<'_, T, R, C>
where
    T: Clone + RealField,
    R: faer::Shape,
    C: faer::Shape,
    u64: TryFrom<C> + TryFrom<R>,
{
    type Owned = Mat<T, R, C>;

    fn into_owned(self) -> Self::Owned {
        self.to_owned()
    }

    fn clone_owned(&self) -> Self::Owned {
        self.to_owned()
    }
    fn ncols(&self) -> Option<u64> {
        self.ncols().try_into().ok()
    }

    fn nrows(&self) -> Option<u64> {
        self.nrows().try_into().ok()
    }
}

impl<T, R, C> Matx<T> for MatMut<'_, T, R, C>
where
    T: Clone + RealField,
    R: faer::Shape,
    C: faer::Shape,
    u64: TryFrom<C> + TryFrom<R>,
{
    type Owned = Mat<T, R, C>;

    fn into_owned(self) -> Self::Owned {
        self.to_owned()
    }

    fn clone_owned(&self) -> Self::Owned {
        self.to_owned()
    }
    fn ncols(&self) -> Option<u64> {
        self.ncols().try_into().ok()
    }

    fn nrows(&self) -> Option<u64> {
        self.nrows().try_into().ok()
    }
}

impl<T, R, C> Ownedx for Mat<T, R, C>
where
    R: Shape,
    C: Shape,
{
}

impl<T, R> Ownedx for Col<T, R> where R: Shape {}

impl<T, R> Colx<T> for Col<T, R>
where
    T: RealField + Copy + Float + AddAssign,
    R: Shape,
    u64: TryFrom<R, Error: std::fmt::Debug>,
{
    type Owned = Self;

    fn enorm(&self) -> T {
        crate::utility::enorm(self.as_col_ref().iter().cloned())
    }

    fn clone_owned(&self) -> Self::Owned {
        Col::<T, R>::clone(self)
    }

    fn into_owned(self) -> Self::Owned {
        self
    }

    fn max(&self) -> Option<T> {
        self.max()
    }

    fn dim(&self) -> Option<u64> {
        self.nrows().try_into().ok()
    }
}

impl<'a, T, R> Colx<T> for ColMut<'a, T, R>
where
    T: RealField + Float + AddAssign + Copy,
    R: Shape,
    u64: TryFrom<R, Error: std::fmt::Debug>,
{
    type Owned = Col<T, R>;

    fn enorm(&self) -> T {
        crate::utility::enorm(self.as_col_ref().iter().copied())
    }

    fn clone_owned(&self) -> Self::Owned {
        self.to_owned()
    }

    fn into_owned(self) -> Self::Owned {
        self.to_owned()
    }

    fn max(&self) -> Option<T> {
        self.max()
    }

    fn dim(&self) -> Option<u64> {
        self.nrows().try_into().ok()
    }
}

impl<'a, T, R> Colx<T> for ColRef<'a, T, R>
where
    T: RealField + Float + AddAssign + Copy,
    R: Shape,
    u64: TryFrom<R, Error: std::fmt::Debug>,
{
    type Owned = Col<T, R>;

    fn enorm(&self) -> T {
        crate::utility::enorm(self.iter().copied())
    }

    fn clone_owned(&self) -> Self::Owned {
        self.to_owned()
    }

    fn into_owned(self) -> Self::Owned {
        self.to_owned()
    }

    fn max(&self) -> Option<T> {
        self.max()
    }

    fn dim(&self) -> Option<u64> {
        self.nrows().try_into().ok()
    }
}

impl<T, R> Scalex<T> for Col<T, R>
where
    T: RealField,
    R: Shape,
{
    fn scale(self, factor: T) -> Self {
        self * Scale(factor)
    }

    fn scale_mut(&mut self, factor: T) {
        *self *= Scale(factor);
    }
}

impl<'a, T, R> Scalex<T> for ColMut<'a, T, R>
where
    T: RealField,
    R: Shape,
{
    fn scale(mut self, factor: T) -> Self {
        self *= Scale(factor);
        self
    }

    fn scale_mut(&mut self, factor: T) {
        *self *= Scale(factor);
    }
}

impl<T, V, R, ColType> Addx<T, V> for ColType
where
    ColType: FaerType + AsColMut<T = T, Rows = R>,
    T: RealField + Copy,
    T: AddAssign,
    V: AsColRef<T = T, Rows = R> + Colx<T>,
    R: Shape,
{
    fn scaled_add(mut self, factor: T, y: &V) -> Option<Self> {
        let this = self.as_col_mut();
        let yref = y.as_col_ref();
        if this.nrows() != yref.nrows() {
            return None;
        }
        //@todo(geo-ant): PERF: is this the most efficient way of implementing
        // this axpy variant?
        #[allow(unused_mut)]
        faer::zip!(this, yref).for_each(|faer::unzip!(mut this, rhs)| *this += factor * *rhs);
        Some(self)
    }
}

impl<T, R, V1, V2> Dotx<T, V2> for V1
where
    V1: FaerType + AsColRef<T = T, Rows = R> + Colx<T>,
    T: RealField,
    R: Shape,
    V2: Colx<T> + AsColRef<T = T, Rows = R>,
{
    fn dot(&self, v: &V2) -> Option<T> {
        let v = v.as_col_ref();
        let this = self.as_col_ref();
        if this.nrows() != v.nrows() {
            return None;
        }

        Some(faer::linalg::matmul::dot::inner_prod(
            this.transpose(),
            faer::Conj::No,
            v,
            faer::Conj::No,
        ))
    }
}

impl<T, R, C, V, M> TrMatVecMulx<T, V> for M
where
    M: AsMatRef<T = T, Rows = R, Cols = C> + FaerType,
    T: RealField + ConstOne + Float + Copy + AddAssign,
    R: Shape,
    C: Shape,
    V: Colx<T> + AsColRef<T = T, Rows = R>,
    u64: TryFrom<C, Error: std::fmt::Debug>,
{
    type Output = Col<T, C>;

    fn tr_mulv(&self, v: &V) -> Option<Self::Output> {
        let m = self.as_mat_ref();
        let v = v.as_col_ref();
        let (transposed_cols, transposed_rows) = (m.nrows(), m.ncols());

        let v_rows = v.nrows();
        if v_rows != transposed_cols {
            return None;
        }

        //@todo(geo-ant) PERFORMANCE: is there a way to have uninitialized
        // values here?
        let mut dest = Col::<T, C>::zeros(transposed_rows);
        faer::linalg::matmul::matmul(
            &mut dest,
            Accum::Replace,
            m.transpose(),
            v,
            T::ONE,
            faer::get_global_parallelism(),
        );
        Some(dest)
    }
}

impl<T, R, C, M, V> TransformedVecNorm<T, V> for M
where
    T: RealField + ConstOne + Float + AddAssign + Copy,
    R: Shape,
    C: Shape,
    M: FaerType + AsMatRef<T = T, Rows = R, Cols = C>,
    V: AsColRef<T = T, Rows = C> + Colx<T>,
    u64: TryFrom<R, Error: std::fmt::Debug>,
{
    fn mulv_enorm(&self, v: &V) -> Option<T> {
        let v = v.as_col_ref();
        let m = self.as_mat_ref();

        if m.ncols() != v.nrows() {
            return None;
        }

        //@todo(geo-ant) PERFORMANCE: is there a way to have uninitialized
        // values here?
        let mut mv = Col::<T, R>::zeros(m.nrows());

        faer::linalg::matmul::matmul(
            &mut mv,
            Accum::Replace,
            m,
            v,
            T::ONE,
            faer::get_global_parallelism(),
        );

        Some(mv.enorm())
    }
}

impl<T, M> ToSvdx<T> for M
where
    T: RealField,
    M: AsMatRef<T = T, Rows = usize, Cols = usize> + FaerType,
{
    type Svd = Svd<T>;

    fn calc_svd(self) -> Option<Self::Svd> {
        Svd::new_thin(self.as_mat_ref()).ok()
    }
}

impl<T, V> Svdx<T, V> for Svd<T>
where
    T: RealField + Float + AddAssign + Copy,
    V: Colx<T> + AsColRef<T = T, Rows = usize>,
{
    type Output = Col<T>;

    fn solve_lsqr(&self, v: &V) -> Option<Self::Output> {
        let x = self.solve_lstsq(v.as_col_ref());
        Some(x)
    }
}

impl<T, R, C, M> ColEnormsx<T> for M
where
    T: RealField + Copy + Float + AddAssign,
    M: AsMatRef<T = T, Rows = R, Cols = C> + FaerType,
    C: Shape,
    R: Shape,
    u64: TryFrom<R, Error: std::fmt::Debug>,
{
    type Output = Col<T>;

    fn column_enorms(&self) -> Self::Output {
        let this = self.as_mat_ref();
        Col::from_iter(this.col_iter().map(|col| col.enorm()))
    }
}

impl<T, V, M, R, C> DiagRightMulx<V> for M
where
    M: FaerType + AsMatMut<T = T, Rows = R, Cols = C>,
    R: Shape,
    C: Shape,
    V: AsColRef<T = T, Rows = C>,
    T: RealField + MulAssign + Copy + Float,
{
    fn mul_diag_right(mut self, diagonal: &V, invert: Invert) -> Option<Self> {
        let this = self.as_mat_mut();
        let diagonal = diagonal.as_col_ref();
        if this.ncols() != diagonal.nrows() {
            return None;
        }

        //@todo(geo-ant) PERF there might be a way to apply a DiagRef to this.transpose()
        // in place, but that doesn't have to be faster necessarily and I'm somehow
        // too dumb to get that to compile, so I'm trying to do the next best thing
        // here.
        match faer::get_global_parallelism() {
            faer::Par::Seq => {
                this.col_iter_mut()
                    .zip(diagonal.iter().copied())
                    .for_each(|(mut col, diag)| {
                        let diag = match invert {
                            crate::Invert::Yes => diag.powi(-1),
                            crate::Invert::No => diag,
                        };
                        col *= Scale(diag);
                    });
            }
            faer::Par::Rayon(_) => {
                this.par_col_iter_mut()
                    .zip(diagonal.par_iter().copied())
                    .for_each(|(mut col, diag)| {
                        let diag = match invert {
                            crate::Invert::Yes => diag.powi(-1),
                            crate::Invert::No => diag,
                        };
                        col *= Scale(diag);
                    });
            }
        }
        Some(self)
    }
}

impl<T, V, R, M> DiagLeftMulx<T, V> for M
where
    M: FaerType + AsColMut<T = T, Rows = R>,
    R: Shape,
    V: AsColRef<T = T, Rows = R>,
    T: RealField + MulAssign + Copy + Float,
{
    fn diag_mul_left(mut self, diagonal: &V, invert: Invert) -> Option<Self> {
        let this = self.as_col_mut();
        let diagonal = diagonal.as_col_ref();

        if this.nrows() != diagonal.nrows() {
            return None;
        }

        match faer::get_global_parallelism() {
            faer::Par::Seq => {
                this.iter_mut()
                    .zip(diagonal.iter().copied())
                    .for_each(|(elem, diag)| {
                        let diag = match invert {
                            crate::Invert::Yes => diag.powi(-1),
                            crate::Invert::No => diag,
                        };
                        *elem *= diag;
                    });
            }
            faer::Par::Rayon(_) => {
                this.par_iter_mut()
                    .zip(diagonal.par_iter().copied())
                    .for_each(|(elem, diag)| {
                        let diag = match invert {
                            crate::Invert::Yes => diag.powi(-1),
                            crate::Invert::No => diag,
                        };
                        *elem *= diag;
                    });
            }
        };
        Some(self)
    }

    fn diag_mul_left_enorm(&self, diagonal: &V) -> Option<T>
    where
        T: AddAssign,
    {
        let this = self.as_col_ref();
        let diagonal = diagonal.as_col_ref();

        if this.nrows() != diagonal.nrows() {
            return None;
        }

        Some(enorm(
            this.iter()
                .copied()
                .zip(diagonal.iter().copied())
                .map(|(elem, diag)| elem * diag),
        ))
    }
}

impl<T, R, V1, V2> MaxScaledDivx<T, V2> for V1
where
    T: RealField + Copy,
    R: Shape,
    V1: FaerType + AsColRef<T = T, Rows = R>,
    V2: FaerType + AsColRef<T = T, Rows = R>,
{
    fn max_scaled_div(&self, s: T, v: &V2) -> Option<T> {
        let this = self.as_col_ref();
        let v = v.as_col_ref();
        faer::zip!(this, v)
            .map(|faer::unzip!(this, rhs)| *this / *rhs)
            .max()
            .map(|val| val / s)
    }
}

impl<T, R, V1, V2> ElementwiseMaxx<V2> for V1
where
    T: RealField + Copy + TotalOrder,
    R: Shape,
    V1: FaerType + AsColMut<T = T, Rows = R>,
    V2: FaerType + AsColRef<T = T, Rows = R>,
{
    fn elementwise_max(mut self, other: &V2) -> Option<Self> {
        let this = self.as_col_mut();
        let other = other.as_col_ref();

        if this.nrows() != other.nrows() {
            return None;
        }

        faer::zip!(this, other).for_each(|faer::unzip!(this, rhs)| {
            let max = match TotalOrder::total_cmp(this, rhs) {
                std::cmp::Ordering::Less => *rhs,
                _ => *this,
            };
            *this = max;
        });
        Some(self)
    }
}

impl<T, R, V> ElementwiseReplaceLeqx<T> for V
where
    T: RealField + Copy + TotalOrder,
    R: Shape,
    V: FaerType + AsColMut<T = T, Rows = R>,
{
    fn replace_if_leq(mut self, threshold: T, replacement: T) -> Self {
        self.as_col_mut().iter_mut().for_each(|elem| {
            if !matches!(elem.total_cmp(&threshold), Ordering::Greater) {
                *elem = replacement;
            }
        });
        self
    }
}
