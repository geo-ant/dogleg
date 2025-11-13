/// Utility trait, defining a couple of constants that the algorithm needs
/// internally. Already implemented for f64 and f32
pub trait MagicConst: num_traits::ConstOne + num_traits::ConstZero {
    /// the value 2
    const TWO: Self;
    /// the value 30
    const THIRTY: Self;
    /// the value 100
    const ONE_HUNDRED: Self;
    /// "point 5" = 0.5 = 1/2 (those weird P<...> are from MINPACK)
    const P5: Self;
}

/// just a macro here to minimize the chance of typos
macro_rules! impl_magic_const {
    ($type:ty) => {
        impl MagicConst for $type {
            const TWO: Self = 2.;
            const THIRTY: Self = 30.;
            const ONE_HUNDRED: Self = 100.;
            const P5: Self = 0.5;
        }
    };
}

impl_magic_const!(f32);
impl_magic_const!(f64);
