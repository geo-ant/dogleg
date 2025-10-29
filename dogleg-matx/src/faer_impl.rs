use crate::{Matx, Ownedx};
use faer::Mat;
use nalgebra::Owned;

impl<T> Matx<T> for Mat<T> {
    type Owned = Self;

    fn into_owned(self) -> Self::Owned {
        self
    }
}

impl<T> Ownedx for Mat<T> {}
