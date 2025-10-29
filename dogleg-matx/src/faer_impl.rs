use crate::{Colx, Matx, Ownedx, Scalex};
use faer::{traits::RealField, Col, ColMut, ColRef, Mat, Scale};

impl<T> Matx<T> for Mat<T>
where
    T: Clone,
{
    type Owned = Self;

    fn into_owned(self) -> Self::Owned {
        self
    }

    fn clone_owned(&self) -> Self::Owned {
        self.clone()
    }
}

impl<T> Ownedx for Mat<T> {}
impl<T> Ownedx for Col<T> {}

impl<T> Colx<T> for Col<T>
where
    T: faer::traits::RealField,
{
    type Owned = Self;

    fn enorm(&self) -> T {
        todo!()
    }

    fn clone_owned(&self) -> Self::Owned {
        Col::<T>::clone(self)
    }

    fn into_owned(self) -> Self::Owned {
        self
    }
}

impl<'a, T> Colx<T> for ColRef<'a, T>
where
    T: faer::traits::RealField,
{
    type Owned = Col<T>;

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

impl<T> Scalex<T> for Col<T>
where
    T: RealField,
{
    fn scale(self, factor: T) -> Self {
        self * Scale(factor)
    }
}

impl<'a, T> Colx<T> for ColMut<'a, T>
where
    T: RealField,
{
    type Owned = Col<T>;

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

impl<'a, T> Scalex<T> for ColMut<'a, T>
where
    T: RealField,
{
    fn scale(mut self, factor: T) -> Self {
        self *= Scale(factor);
        self
    }
}
