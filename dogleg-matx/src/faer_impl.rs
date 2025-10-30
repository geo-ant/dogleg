use crate::{
    Addx, Colx, Dotx, Matx, Ownedx, Scalex, Svdx, ToSvdx, TrMatVecMulx, TransformedVecNorm,
};
use faer::col::AsColMut;
use faer::linalg::solvers::Svd;
use faer::mat::AsMatRef;
use faer::prelude::SolveLstsq;
use faer::{col::AsColRef, traits::RealField, Col, ColMut, ColRef, Mat, Scale};
use faer::{Accum, MatMut, MatRef, Shape};
use num_traits::ConstOne;
use std::ops::AddAssign;

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
{
    type Owned = Self;

    fn into_owned(self) -> Self::Owned {
        self
    }

    fn clone_owned(&self) -> Self::Owned {
        self.clone()
    }
}

impl<T, R, C> Matx<T> for MatRef<'_, T, R, C>
where
    T: Clone + RealField,
    R: faer::Shape,
    C: faer::Shape,
{
    type Owned = Mat<T, R, C>;

    fn into_owned(self) -> Self::Owned {
        self.to_owned()
    }

    fn clone_owned(&self) -> Self::Owned {
        self.to_owned()
    }
}

impl<T, R, C> Matx<T> for MatMut<'_, T, R, C>
where
    T: Clone + RealField,
    R: faer::Shape,
    C: faer::Shape,
{
    type Owned = Mat<T, R, C>;

    fn into_owned(self) -> Self::Owned {
        self.to_owned()
    }

    fn clone_owned(&self) -> Self::Owned {
        self.to_owned()
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
    T: faer::traits::RealField,
    R: Shape,
{
    type Owned = Self;

    fn enorm(&self) -> T {
        todo!()
    }

    fn clone_owned(&self) -> Self::Owned {
        Col::<T, R>::clone(self)
    }

    fn into_owned(self) -> Self::Owned {
        self
    }
}

impl<'a, T, R> Colx<T> for ColMut<'a, T, R>
where
    T: RealField,
    R: Shape,
{
    type Owned = Col<T, R>;

    fn enorm(&self) -> T {
        todo!()
    }

    fn clone_owned(&self) -> Self::Owned {
        self.to_owned()
    }

    fn into_owned(self) -> Self::Owned {
        self.to_owned()
    }
}

impl<'a, T, R> Colx<T> for ColRef<'a, T, R>
where
    T: faer::traits::RealField,
    R: Shape,
{
    type Owned = Col<T, R>;

    fn enorm(&self) -> T {
        todo!()
    }

    fn clone_owned(&self) -> Self::Owned {
        self.to_owned()
    }

    fn into_owned(self) -> Self::Owned {
        self.to_owned()
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
    T: RealField + ConstOne,
    R: Shape,
    C: Shape,
    V: Colx<T> + AsColRef<T = T, Rows = R>,
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
    T: RealField + ConstOne,
    R: Shape,
    C: Shape,
    M: FaerType + AsMatRef<T = T, Rows = R, Cols = C>,
    V: AsColRef<T = T, Rows = C> + Colx<T>,
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
    T: RealField,
    V: Colx<T> + AsColRef<T = T, Rows = usize>,
{
    type Output = Col<T>;

    fn solve_lsqr(&self, v: &V) -> Option<Self::Output> {
        let x = self.solve_lstsq(v.as_col_ref());
        Some(x)
    }
}
