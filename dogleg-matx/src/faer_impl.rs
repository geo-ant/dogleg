use std::ops::AddAssign;

use crate::{Addx, Colx, Matx, Ownedx, Scalex};
use faer::Shape;
use faer::{col::AsColRef, traits::RealField, Col, ColMut, ColRef, Mat, Scale};

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

impl<T, V, R> Addx<T, V> for Col<T, R>
where
    T: RealField + Copy,
    T: AddAssign,
    V: AsColRef<T = T, Rows = R> + Colx<T>,
    R: Shape,
{
    fn scaled_add(mut self, factor: T, y: &V) -> Option<Self> {
        let yref = y.as_col_ref();
        if self.nrows() != yref.nrows() {
            return None;
        }
        //@todo(geo-ant): PERF: is this the most efficient way of implementing
        // this axpy variant?
        #[allow(unused_mut)]
        faer::zip!(&mut self, yref).for_each(|faer::unzip!(mut this, rhs)| *this += factor * *rhs);
        Some(self)
    }
}
