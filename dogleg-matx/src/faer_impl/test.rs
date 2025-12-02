use crate::{col_assert_relative_eq, Addx, Colx, Dotx, Matx, Scalex, TrMatVecMulx};
use approx::assert_relative_eq;
use faer::mat::AsMatRef;

#[test]
#[should_panic]
fn col_assert_relative_eq_test_should_panic() {
    col_assert_relative_eq!(faer::col!(1., 2.), faer::col!(1.000000001, 2.));
}

#[test]
#[should_panic]
fn col_assert_relative_eq_test_should_panic2() {
    col_assert_relative_eq!(faer::col!(1., 2.), faer::col!(1., 2.000000001));
}

#[test]
fn col_assert_relative_eq_test_works() {
    col_assert_relative_eq!(faer::col!(1., 2.), faer::col!(1., 2.), epsilon = 1e-10);
    col_assert_relative_eq!(faer::col!(1., 2.), faer::col!(1.1, 1.9), epsilon = 0.2);
}

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
// @todo(geo-ant): also make proptests
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

#[test]
fn vector_addx() {
    let v1 = faer::col!(123., 0.1, 1.337);
    let v2 = faer::col!(1.23, 0.005, 9.84);

    col_assert_relative_eq!(
        Addx::scaled_add(v1.clone(), -1., &v1).unwrap(),
        &faer::col!(0., 0., 0.),
        epsilon = 1e-10
    );

    col_assert_relative_eq!(
        Addx::scaled_add(v2.clone(), -1., &v2).unwrap(),
        faer::col!(0., 0., 0.),
        epsilon = 1e-10
    );

    col_assert_relative_eq!(
        Addx::scaled_add(v1.clone(), -2., &v2).unwrap(),
        &v1 - &v2 * 2.,
        epsilon = 1e-10
    );

    col_assert_relative_eq!(
        Addx::add(v1.clone(), &v2).unwrap(),
        &v1 + &v2,
        epsilon = 1e-10
    );

    assert!(Addx::scaled_add(v1, -2., &faer::col!(1., 2.)).is_none());
}

#[test]
fn vector_dotx() {
    let svec1 = faer::col!(123., 0.1, 1.337);
    let svec2 = faer::col!(-10., 4.234, -1234.);

    assert_relative_eq!(
        Dotx::dot(&svec1, &svec2).unwrap(),
        -123. * 10. + 0.1 * 4.234 - 1.337 * 1234.,
        epsilon = 1e-10
    );
    assert!(Dotx::dot(&svec1, &faer::col!(1., 2.)).is_none());
}

#[test]
fn tr_mat_mul_vec() {
    let v = faer::col![28.2, 0.1234];
    let mat = faer::mat![[4., 99., 0.1, 0.9], [8., 0.5, 3455., 777.]];
    col_assert_relative_eq!(
        TrMatVecMulx::tr_mulv(&mat, &v).unwrap(),
        mat.transpose() * v,
        epsilon = 1e-10,
    );

    assert!(TrMatVecMulx::tr_mulv(&mat, &faer::col![1.]).is_none());
}
