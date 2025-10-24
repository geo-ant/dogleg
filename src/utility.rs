use nalgebra::{Dim, RealField, Storage, Vector, U1};
use num_traits::Float;

#[inline]
#[cold]
fn cold() {}

#[inline]
fn likely(b: bool) -> bool {
    if !b {
        cold()
    }
    b
}

#[inline]
fn unlikely(b: bool) -> bool {
    if b {
        cold()
    }
    b
}

#[inline]
#[allow(clippy::unreadable_literal)]
/// Machine epsilon
/// Directly taken from the `levenberg_marquardt` crate, which in turn translated it
/// from the FORTRAN implementation in MINPACK.
pub(crate) fn epsmch<F: RealField>() -> F {
    F::default_epsilon()
}

#[inline]
#[allow(clippy::unreadable_literal)]
/// A very large number
/// Directly taken from the `levenberg_marquardt` crate, which in turn translated it
/// from the FORTRAN implementation in MINPACK.
pub(crate) fn giant<F: Float>() -> F {
    F::max_value()
}

#[inline]
#[allow(clippy::unreadable_literal)]
/// A very small number
/// Directly taken from the `levenberg_marquardt` crate, which in turn translated it
/// from the FORTRAN implementation in MINPACK.
pub(crate) fn dwarf<F: Float>() -> F {
    F::min_positive_value()
}

#[inline]
/// Numerically more stable euclidean_norm of a vector.
///
/// Directly taken from the `levenberg_marquardt` crate, which in turn translated it
/// from the FORTRAN implementation in MINPACK.
pub(crate) fn enorm<F, N, VS>(v: &Vector<F, N, VS>) -> F
where
    F: nalgebra::RealField + Float + Copy,
    N: Dim,
    VS: Storage<F, N, U1>,
{
    let mut s1 = F::zero();
    let mut s2 = F::zero();
    let mut s3 = F::zero();
    let mut x1max = F::zero();
    let mut x3max = F::zero();
    let agiant = Float::sqrt(giant::<F>());
    let rdwarf = Float::sqrt(dwarf());
    for xi in v.iter() {
        let xabs = xi.abs();
        if unlikely(xabs.is_nan()) {
            return xabs;
        }
        if unlikely(xabs >= agiant || xabs <= rdwarf) {
            if xabs > rdwarf {
                // sum for large components
                if xabs > x1max {
                    s1 = F::one() + s1 * Float::powi(x1max / xabs, 2);
                    x1max = xabs;
                } else {
                    s1 += Float::powi(xabs / x1max, 2);
                }
            } else {
                // sum for small components
                if xabs > x3max {
                    s3 = F::one() + s3 * Float::powi(x3max / xabs, 2);
                    x3max = xabs;
                } else if xabs != F::zero() {
                    s3 += Float::powi(xabs / x3max, 2);
                }
            }
        } else {
            s2 += xabs * xabs;
        }
    }

    if unlikely(!s1.is_zero()) {
        x1max * Float::sqrt(s1 + (s2 / x1max) / x1max)
    } else if likely(!s2.is_zero()) {
        Float::sqrt(if likely(s2 >= x3max) {
            s2 * (F::one() + (x3max / s2) * (x3max * s3))
        } else {
            x3max * ((s2 / x3max) + (x3max * s3))
        })
    } else {
        x3max * Float::sqrt(s3)
    }
}
