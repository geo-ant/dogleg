/// Utility trait, defining a couple of constants that the algorithm needs
/// internally. Already implemented for f64 and f32
pub trait MagicConst: num_traits::ConstOne + num_traits::ConstZero {
    /// the value 2
    const TWO: Self;
    /// the value 3
    const THREE: Self;
    /// the value 10
    const TEN: Self;
    /// the value 30
    const THIRTY: Self;
    /// the value 100
    const ONE_HUNDRED: Self;
    /// the value 1E16
    const ONE_E16: Self;
    /// (those weird P<...> are from MINPACK)
    /// "point 75" = 0.75 = 3/4
    const P75: Self;
    /// "point 5" = 0.5 = 1/2
    const P5: Self;
    /// "point 25" = 0.25 = 1/4
    const P25: Self;
    /// "point 1" = 0.1
    const P1: Self;
    /// "point 001 = 1e-3"
    const P001: Self;
    /// "point 0001" = 0.
    const P0001: Self;
    /// machine epsilon
    const EPSMCH: Self;
    ///@todo(HACK) remove
    const EMINUS8: Self;
}

/// just a macro here to minimize the chance of typos
macro_rules! impl_magic_const {
    ($type:ty) => {
        impl MagicConst for $type {
            const TWO: Self = 2.;
            const THREE: Self = 3.;
            const TEN: Self = 10.;
            const THIRTY: Self = 30.;
            const ONE_HUNDRED: Self = 100.;
            const ONE_E16: Self = 1e16;
            const P75: Self = 0.75;
            const P5: Self = 0.5;
            const P25: Self = 0.25;
            const P1: Self = 0.1;
            const P001: Self = 1e-3;
            const P0001: Self = 1e-4;
            const EPSMCH: Self = Self::EPSILON;
            const EMINUS8: Self = 1e-8;
        }
    };
}

impl_magic_const!(f32);
impl_magic_const!(f64);
