use crate::{
    Addx, ColEnormsx, Colx, DiagLeftMulx, DiagRightMulx, Dotx, ElementwiseMaxx,
    ElementwiseReplaceLeqx, Matx, MaxAbsx, Scalex, Svdx, ToSvdx, TrMatVecMulx, TransformedVecNorm,
};
use approx::assert_relative_eq;
use nalgebra::{DMatrix, SMatrix, Vector};

macro_rules! sdmat {
    ( $($($elem:expr),*);*) => {
        (nalgebra::matrix![$($($elem),*);*], nalgebra::dmatrix![$($($elem),*);*])
    };
}

macro_rules! sdvec {
    ($($elem:expr),* $(,)?) => {
        (nalgebra::vector![$($elem),*], nalgebra::dvector![$($elem),*])
    };
}

#[test]
fn matx_base_functions_for_smat_and_dmatrix() {
    let smat = nalgebra::matrix![
        1.,2.;
        3.,4.;
        5.,6.];
    let dmat = nalgebra::dmatrix![
        1.,2.;
        3.,4.;
        5.,6.];

    assert_eq!(Matx::<_>::into_owned(smat.clone()), smat);
    assert_eq!(Matx::<_>::clone_owned(&smat), smat);
    assert_eq!(Matx::<_>::ncols(&smat), Some(2));
    assert_eq!(Matx::<_>::nrows(&smat), Some(3));

    assert_eq!(Matx::<_>::into_owned(dmat.clone()), dmat);
    assert_eq!(Matx::<_>::clone_owned(&dmat), dmat);
    assert_eq!(Matx::<_>::ncols(&dmat), Some(2));
    assert_eq!(Matx::<_>::nrows(&dmat), Some(3));
}

#[test]
// we test everything except enorm() here
fn colx_base_functions_for_svec_and_dvector() {
    let svec = nalgebra::vector![1., -4., 2.];
    let dvec = nalgebra::dvector![1., -4., 2.];

    assert_eq!(Colx::<_>::into_owned(svec.clone()), svec);
    assert_eq!(Colx::<_>::clone_owned(&svec), svec);
    assert_eq!(Colx::<_>::max(&svec), Some(2.));
    assert_eq!(Colx::<_>::dim(&svec), Some(3));
    assert_eq!(Colx::<_>::max_absolute(&svec), Some(4.));

    assert_eq!(Colx::<_>::into_owned(dvec.clone()), dvec);
    assert_eq!(Colx::<_>::clone_owned(&dvec), dvec);
    assert_eq!(Colx::<_>::max(&dvec), Some(2.));
    assert_eq!(Colx::<_>::dim(&dvec), Some(3));
    assert_eq!(Colx::<_>::max_absolute(&dvec), Some(4.));
}

#[test]
// @todo(geo-ant): we need proptests for this, but this serves as a reasonable
// smoketest plus one.
fn vector_enorm() {
    let svec = nalgebra::vector!(123., 0.1, 1.337);
    let dvec = nalgebra::dvector!(123., 0.1, 1.337);

    assert_relative_eq!(Colx::<_>::enorm(&svec), svec.norm(), epsilon = 1e-10);
    assert_relative_eq!(Colx::<_>::enorm(&dvec), dvec.norm(), epsilon = 1e-10);
}

#[test]
fn vector_scalex() {
    let svec = nalgebra::vector!(123., 0.1, 1.337);
    let dvec = nalgebra::dvector!(123., 0.1, 1.337);

    let mut sscaled = svec.clone();
    Scalex::scale_mut(&mut sscaled, 5.1);
    assert_eq!(sscaled, svec * 5.1);
    assert_eq!(sscaled, Scalex::scale(svec, 5.1));

    let mut dscaled = dvec.clone();
    Scalex::scale_mut(&mut dscaled, 5.1);
    assert_eq!(dscaled, &dvec * 5.1);
    assert_eq!(dscaled, Scalex::scale(dvec, 5.1));
}

#[test]
fn vector_addx() {
    let svec = nalgebra::vector!(123., 0.1, 1.337);
    let dvec = nalgebra::dvector!(1.23, 0.005, 9.84);

    assert_relative_eq!(
        Addx::scaled_add(svec, -1., &svec).unwrap(),
        nalgebra::vector!(0., 0., 0.),
        epsilon = 1e-10
    );

    assert_relative_eq!(
        Addx::scaled_add(dvec.clone(), -1., &dvec).unwrap(),
        nalgebra::dvector!(0., 0., 0.),
        epsilon = 1e-10
    );

    assert_relative_eq!(
        Addx::scaled_add(svec, -2., &dvec).unwrap(),
        svec - &dvec * 2.,
        epsilon = 1e-10
    );

    assert_relative_eq!(
        Addx::add(svec, &dvec).unwrap(),
        svec + dvec,
        epsilon = 1e-10
    );

    assert!(Addx::scaled_add(svec, -2., &nalgebra::dvector!(1., 2.)).is_none());
}

#[test]
fn vector_dotx() {
    let svec1 = nalgebra::vector!(123., 0.1, 1.337);
    let svec2 = nalgebra::vector!(-10., 4.234, -1234.);
    let dvec1 = nalgebra::dvector!(1.23, 0.005, 9.84);
    let dvec2 = nalgebra::dvector!(2.23, -10.005, 1995.);

    assert_relative_eq!(
        Dotx::dot(&svec1, &svec2).unwrap(),
        Vector::dot(&svec1, &svec2),
        epsilon = 1e-10
    );
    assert_relative_eq!(
        Dotx::dot(&dvec1, &dvec2).unwrap(),
        Vector::dot(&dvec1, &dvec2),
        epsilon = 1e-10
    );

    assert!(Dotx::dot(&dvec1, &nalgebra::dvector!(1., 2.)).is_none());
    assert!(Dotx::dot(&svec1, &nalgebra::dvector!(1., 2.)).is_none());
}

#[test]
fn tr_mat_mul_vec() {
    let svec = nalgebra::vector![28.2, 0.1234];
    let smat = nalgebra::matrix![
        4., 99., 0.1,0.9;
        8.,0.5, 3455.,777.;
    ];
    assert_relative_eq!(
        TrMatVecMulx::tr_mulv(&smat, &svec).unwrap(),
        smat.transpose() * svec,
        epsilon = 1e-10,
    );

    let dvec = nalgebra::dvector![28.2, 0.1234];
    let dmat = nalgebra::dmatrix![
        4., 99., 0.1,0.9;
        8.,0.5, 3455.,777.;
    ];
    assert_relative_eq!(
        TrMatVecMulx::tr_mulv(&dmat, &dvec).unwrap(),
        dmat.transpose() * dvec.clone(),
        epsilon = 1e-10,
    );

    assert_relative_eq!(
        TrMatVecMulx::tr_mulv(&smat, &dvec).unwrap(),
        smat.transpose() * dvec.clone(),
        epsilon = 1e-10,
    );

    assert_relative_eq!(
        TrMatVecMulx::tr_mulv(&dmat, &svec).unwrap(),
        dmat.transpose() * dvec,
        epsilon = 1e-10,
    );

    assert!(TrMatVecMulx::tr_mulv(&dmat, &nalgebra::vector![1.]).is_none());
}

#[test]
fn transformed_vec_norm_for_matrix() {
    let (svec, dvec) = sdvec![3., 1919.];
    let (smat, dmat) = sdmat![
        999.88, 0.1;
        1.3,5.;
        12.34,0.123
    ];

    assert_relative_eq!(
        TransformedVecNorm::mulv_enorm(&smat, &svec).unwrap(),
        (smat * svec).enorm(),
        epsilon = 1e-10
    );

    assert_relative_eq!(
        TransformedVecNorm::mulv_enorm(&dmat, &svec).unwrap(),
        (smat * svec).enorm(),
        epsilon = 1e-10
    );

    assert_relative_eq!(
        TransformedVecNorm::mulv_enorm(&dmat, &dvec).unwrap(),
        (smat * svec).enorm(),
        epsilon = 1e-10
    );

    assert_relative_eq!(
        TransformedVecNorm::mulv_enorm(&smat, &dvec).unwrap(),
        (smat * svec).enorm(),
        epsilon = 1e-10
    );

    assert!(TransformedVecNorm::mulv_enorm(&smat, &nalgebra::dvector![1.]).is_none());
    assert!(TransformedVecNorm::mulv_enorm(&dmat, &nalgebra::dvector![1.]).is_none());
}

#[test]
fn matrix_to_svd_and_solve_lsqr() {
    let (svec, dvec) = sdvec![3., 1919., 0.1];
    let (smat, dmat) = sdmat![
        999.88, 0.1;
        1.3,5.;
        12.34,0.123
    ];

    let ssvd = ToSvdx::calc_svd(smat).unwrap();
    let dsvd = ToSvdx::calc_svd(dmat.clone()).unwrap();

    assert_relative_eq!(
        Svdx::solve_lsqr(&ssvd, &svec).unwrap(),
        smat.svd(true, true).solve(&svec, f64::EPSILON).unwrap()
    );

    assert_relative_eq!(
        Svdx::solve_lsqr(&dsvd, &dvec).unwrap(),
        dmat.svd(true, true).solve(&dvec, f64::EPSILON).unwrap()
    );

    todo!("test solve regularized lsqr");
}

#[test]
fn matrix_col_enorms() {
    let (smat, dmat) = sdmat![
        1.,4.;
        2.,5.;
        3.,6.;
    ];

    let (sexpected, dexpected) = sdvec!(14_f64.sqrt(), 77_f64.sqrt());

    assert_relative_eq!(ColEnormsx::column_enorms(&smat), sexpected);
    assert_relative_eq!(ColEnormsx::column_enorms(&dmat), dexpected);

    let sexpected = sexpected.map(|x| 1. / (1. + x));
    let dexpected = dexpected.map(|x| 1. / (1. + x));

    assert_relative_eq!(ColEnormsx::damped_inverse_column_enorms(&smat), sexpected);
    assert_relative_eq!(ColEnormsx::damped_inverse_column_enorms(&dmat), dexpected);
}

#[test]
fn diag_right_mul_for_matrix() {
    let (smat, dmat) = sdmat![
        2.  , 0.1 , 99.;
        91.8, 2.  , 444.4;
        0.66, 123., 9.;
        6.  , 77. , 0.18;
    ];

    let (sdiag, ddiag) = sdvec![0.4, 33., -18.];
    let sdiagmat = SMatrix::from_diagonal(&sdiag);
    let ddiagmat = DMatrix::from_diagonal(&ddiag);

    assert_relative_eq!(
        DiagRightMulx::mul_diag_right(smat, &sdiag, crate::Invert::No).unwrap(),
        smat * sdiagmat,
        epsilon = 1e-10
    );
    assert_relative_eq!(
        DiagRightMulx::mul_diag_right(smat, &sdiag, crate::Invert::Yes).unwrap(),
        smat * sdiagmat.try_inverse().unwrap(),
        epsilon = 1e-10
    );

    assert_relative_eq!(
        DiagRightMulx::mul_diag_right(dmat.clone(), &ddiag, crate::Invert::No).unwrap(),
        dmat.clone() * ddiagmat.clone(),
        epsilon = 1e-10
    );

    assert_relative_eq!(
        DiagRightMulx::mul_diag_right(dmat.clone(), &ddiag, crate::Invert::Yes).unwrap(),
        dmat.clone() * ddiagmat.try_inverse().unwrap(),
        epsilon = 1e-10
    );

    assert!(DiagRightMulx::mul_diag_right(
        dmat.clone(),
        &nalgebra::dvector![1.],
        crate::Invert::Yes
    )
    .is_none());

    assert!(
        DiagRightMulx::mul_diag_right(smat, &nalgebra::dvector![1.], crate::Invert::Yes).is_none()
    );
    assert!(
        DiagRightMulx::mul_diag_right(dmat, &nalgebra::dvector![1.], crate::Invert::Yes).is_none()
    );
}

#[test]
fn diag_left_mul() {
    let (svec, dvec) = sdvec![2., -0.1, 99., -0.1];
    let (sdiag, ddiag) = sdvec![0.4, 33., -18., -77.6];
    let sdiagmat = SMatrix::from_diagonal(&sdiag);
    let ddiagmat = DMatrix::from_diagonal(&ddiag);

    assert_relative_eq!(
        DiagLeftMulx::diag_mul_left_enorm(&svec, &sdiag).unwrap(),
        (sdiagmat * svec).enorm(),
        epsilon = 1e-10
    );

    assert_relative_eq!(
        DiagLeftMulx::diag_mul_left(svec, &sdiag, crate::Invert::No).unwrap(),
        sdiagmat * svec,
        epsilon = 1e-10
    );

    assert_relative_eq!(
        DiagLeftMulx::diag_mul_left(svec, &sdiag, crate::Invert::Yes).unwrap(),
        sdiagmat.try_inverse().unwrap() * svec,
        epsilon = 1e-10
    );

    assert_relative_eq!(
        DiagLeftMulx::diag_mul_left_enorm(&dvec, &sdiag).unwrap(),
        (sdiagmat * &dvec).enorm(),
        epsilon = 1e-10
    );

    assert_relative_eq!(
        DiagLeftMulx::diag_mul_left_enorm(&dvec.clone(), &ddiag).unwrap(),
        (&ddiagmat * &dvec).enorm(),
        epsilon = 1e-10
    );

    assert_relative_eq!(
        DiagLeftMulx::diag_mul_left(dvec.clone(), &ddiag, crate::Invert::No).unwrap(),
        ddiagmat.clone() * dvec.clone(),
        epsilon = 1e-10
    );

    assert_relative_eq!(
        DiagLeftMulx::diag_mul_left(dvec.clone(), &ddiag, crate::Invert::Yes).unwrap(),
        ddiagmat.try_inverse().unwrap() * dvec.clone(),
        epsilon = 1e-10
    );

    assert!(
        DiagLeftMulx::diag_mul_left(svec, &nalgebra::dvector![1.], crate::Invert::Yes).is_none()
    );

    assert!(
        DiagLeftMulx::diag_mul_left(dvec, &nalgebra::dvector![1.], crate::Invert::Yes).is_none()
    );
}

#[test]
fn max_scaled_div_for_vector() {
    let (svec1, dvec1) = sdvec![2., 3., 4.];
    let (svec2, dvec2) = sdvec![8., 6., 100.];
    let scale = 2.;

    assert_eq!(
        MaxAbsx::max_abs_scaled_div_elem(&svec1, scale, &svec2).unwrap(),
        (3. / 12.)
    );
    assert_eq!(
        MaxAbsx::max_abs_scaled_div_elem(&dvec1, scale, &dvec2).unwrap(),
        (3. / 12.)
    );

    let (svec1, dvec1) = sdvec![2., -3., 4.];
    let (svec2, dvec2) = sdvec![8., 6., 100.];
    let scale = 2.;

    assert_eq!(
        MaxAbsx::max_abs_scaled_div_elem(&svec1, scale, &svec2).unwrap(),
        (3. / 12.)
    );
    assert_eq!(
        MaxAbsx::max_abs_scaled_div_elem(&dvec1, scale, &dvec2).unwrap(),
        (3. / 12.)
    );
}

#[test]
fn elementwise_max_for_vector() {
    let (svec1, dvec1) = sdvec![-100., 0.1, 2., 99.1];
    let (svec2, dvec2) = sdvec![-101., 0.2, 1.9, 99.2];
    let (sexpected, dexpected) = sdvec![-100., 0.2, 2., 99.2];

    assert_eq!(
        ElementwiseMaxx::elementwise_max(svec1, &svec2).unwrap(),
        sexpected
    );
    assert_eq!(
        ElementwiseMaxx::elementwise_max(dvec1, &dvec2).unwrap(),
        dexpected
    );
    assert!(ElementwiseMaxx::elementwise_max(dvec2, &nalgebra::vector![1.]).is_none());
    assert!(ElementwiseMaxx::elementwise_max(svec2, &nalgebra::dvector![1.]).is_none());
}

#[test]
fn elementwise_replace_if_leq_for_vector() {
    let (svec, dvec) = sdvec![5.2, -100.1, 2., 99.1];
    let threshold = 5.2;
    let replacement = 123.;
    let (sexpected, dexpected) = sdvec![123., 123., 123., 99.1];

    assert_eq!(
        ElementwiseReplaceLeqx::replace_if_leq(svec, threshold, replacement),
        sexpected
    );
    assert_eq!(
        ElementwiseReplaceLeqx::replace_if_leq(dvec, threshold, replacement),
        dexpected
    );

    let (svec, dvec) = sdvec![5.2, -100.1, 2., -30., 49., 99.1];
    let (sexpected, dexpected) = sdvec![5.2, -50., 2., -30., 49., 55.];

    assert_eq!(ElementwiseReplaceLeqx::clamp(svec, -50., 55.), sexpected);
    assert_eq!(ElementwiseReplaceLeqx::clamp(dvec, -50., 55.), dexpected);
}
