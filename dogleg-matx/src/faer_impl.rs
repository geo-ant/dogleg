use crate::{Matx, Ownedx};
use faer::Mat;

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
