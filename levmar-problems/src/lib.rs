//! Test crate that provides the test problems from the levenberg-marquardt
//! crate to the rest of the crates in this test suite. All files in this
//! crate were copied (and lightly edited in parts) from the `levenberg-marquardt` crate.
//! See https://github.com/rust-cv/levenberg-marquardt
//! and https://crates.io/crates/levenberg-marquardt/
//!
//! ORIGINAL COPYRIGHT NOTICE:
//!
//! ------------------------------------------------------------------------------
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
//!
//! ------------------------------------------------------------------------------

pub mod assertions;
pub mod problems;
pub mod utils;

pub use approx::assert_relative_eq;
pub use approx::relative_eq;
