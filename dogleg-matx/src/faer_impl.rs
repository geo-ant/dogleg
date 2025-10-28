use crate::{Matx, Ownedx};
use faer::Mat;
use nalgebra::Owned;

impl<T> Matx<T> for Mat<T> {
    type Owned = Self;
}

impl<T> Ownedx for Mat<T> {}
