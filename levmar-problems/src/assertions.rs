pub use approx::assert_relative_eq;
pub use approx::relative_eq;
pub use assert2::check;

/// helper macro for floating point equality comparisons
#[macro_export]
macro_rules! assert_fp_eq {
    ($given:expr, $expected:expr) => {
        // NOTE(geo-ant): this 1e-12 needs to be kept in sync with
        // the value in the check_fp_eq! macro.
        $crate::assertions::assert_relative_eq!($given, $expected, epsilon = 1e-12)
    };

    ($given:expr, $expected:expr, epsilon = $ep:literal) => {
        $crate::assertions::assert_relative_eq!($given, $expected, epsilon = $ep)
    };

    ($given:expr, $expected:expr, $ep:expr) => {
        $crate::assertions::assert_relative_eq!($given, $expected, epsilon = $ep)
    };
}

// /// helper macro for floating point equality comparisons that uses the assert2::check
// /// macro under the hood for delayed panics.
// #[macro_export]
// macro_rules! check_fp_eq {
//     ($given:expr, $expected:expr) => {
//         // NOTE(geo-ant): this 1e-12 needs to be kept in sync with
//         // the value in the assert_fp_eq! macro.
//         let guard = if !$crate::relative_eq!($given, $expected, epsilon = 1e-12) {
//             $crate::assertions::check!($given == $expected)
//         } else {
//             $crate::assertions::check!($given == $given);
//         };
//     };

//     ($given:expr, $expected:expr, epsilon = $ep:literal) => {
//         if !$crate::relative_eq!($given, $expected, epsilon = $ep) {
//             $crate::assertions::check!($given == $expected);
//         }
//     };

//     ($given:expr, $expected:expr, $ep:expr) => {
//         if !$crate::relative_eq!($given, $expected, epsilon = $ep) {
//             $crate::assertions::check!($given == $expected);
//         }
//     };
// }
