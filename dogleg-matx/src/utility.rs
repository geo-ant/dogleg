//! utility code that is mostly directly cribbed from the `levenberg-marquardt`
//! crate, who ported it from MINPACK.
//!
//! Original license of the `levenberg-marquardt` crate is:
//!
//! MIT License
//!
//! Copyright (c) 2020 rust-cv
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.

use num_traits::Float;
use std::ops::AddAssign;

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
/// A very large number
/// Directly taken from the `levenberg_marquardt` crate, which in turn translated it
/// from the FORTRAN implementation in MINPACK.
pub fn giant<T: Float>() -> T {
    T::max_value()
}

#[inline]
#[allow(clippy::unreadable_literal)]
/// A very small number
/// Directly taken from the `levenberg_marquardt` crate, which in turn translated it
/// from the FORTRAN implementation in MINPACK.
pub fn dwarf<T: Float>() -> T {
    T::min_positive_value()
}

/// Numerically more stable euclidean_norm of a vector.
///
/// Directly taken from the `levenberg_marquardt` crate, which in turn translated it
/// from the FORTRAN implementation in MINPACK.
#[inline]
pub(crate) fn enorm<T>(v: impl Iterator<Item = T>) -> T
where
    T: Float + Copy + AddAssign,
{
    let mut s1 = T::zero();
    let mut s2 = T::zero();
    let mut s3 = T::zero();
    let mut x1max = T::zero();
    let mut x3max = T::zero();
    let agiant = Float::sqrt(giant::<T>());
    let rdwarf = Float::sqrt(dwarf());
    for xi in v {
        let xabs = xi.abs();
        if unlikely(xabs.is_nan()) {
            return xabs;
        }
        if unlikely(xabs >= agiant || xabs <= rdwarf) {
            if xabs > rdwarf {
                // sum for large components
                if xabs > x1max {
                    s1 = T::one() + s1 * Float::powi(x1max / xabs, 2);
                    x1max = xabs;
                } else {
                    s1 += Float::powi(xabs / x1max, 2);
                }
            } else {
                // sum for small components
                if xabs > x3max {
                    s3 = T::one() + s3 * Float::powi(x3max / xabs, 2);
                    x3max = xabs;
                } else if xabs != T::zero() {
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
            s2 * (T::one() + (x3max / s2) * (x3max * s3))
        } else {
            x3max * ((s2 / x3max) + (x3max * s3))
        })
    } else {
        x3max * Float::sqrt(s3)
    }
}
