/// Utility trait, defining a couple of constants that the algorithm needs
/// internally. Already implemented for f64 and f32
pub trait MagicConst: num_traits::ConstOne + num_traits::ConstZero {
    /// the value 30
    const THIRTY: Self;
    /// the value 1E+2 = 100
    const ONE_E2: Self;
}

impl MagicConst for f32 {
    const THIRTY: Self = 30.;
    const ONE_E2: Self = 100.;
}

impl MagicConst for f64 {
    const THIRTY: Self = 30.;
    const ONE_E2: Self = 100.;
}
