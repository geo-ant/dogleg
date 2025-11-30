use crate::{Addx, Colx, Matx, Scalex};
use approx::{assert_abs_diff_eq, assert_relative_eq};
use faer::mat::AsMatRef;

#[test]
fn matx_base_functions() {
    let mat = faer::mat![[1., 2.], [3., 4.], [5., 6.]];

    assert_eq!(Matx::into_owned(mat.clone()), mat);
    assert_eq!(Matx::into_owned(mat.as_mat_ref()), mat);
    assert_eq!(Matx::clone_owned(&mat), mat);
    assert_eq!(Matx::ncols(&mat), Some(2));
    assert_eq!(Matx::nrows(&mat), Some(3));
}

#[test]
// we test everything except enorm() here
fn colx_base_functions_for_svec_and_dvector() {
    let vec = faer::col![1., 4., 2.];

    assert_eq!(Colx::into_owned(vec.clone()), vec);
    assert_eq!(Colx::clone_owned(&vec), vec);
    assert_eq!(Colx::max(&vec), Some(4.));
    assert_eq!(Colx::dim(&vec), Some(3));
}

#[test]
// @todo(geo-ant): we need proptests for this, but this serves as a reasonable
// smoketest plus one.
fn vector_enorm() {
    let v = faer::col!(123., 0.1, 1.337);

    assert_relative_eq!(Colx::<_>::enorm(&v), v.norm_l2(), epsilon = 1e-10);
}

#[test]
fn vector_scalex() {
    let v = faer::col!(123., 0.1, 1.337);

    let mut scaled = v.clone();
    Scalex::scale_mut(&mut scaled, 5.1);
    assert_eq!(scaled, 5.1 * v.clone());
    assert_eq!(scaled, Scalex::scale(v, 5.1));
}
