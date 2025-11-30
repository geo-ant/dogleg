use crate::{Colx, Matx};
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
