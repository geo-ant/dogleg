use approx::assert_relative_eq;
use nalgebra::Vector;

use crate::{Addx, Colx, Dotx, Matx, OwnedMatx, Scalex};

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
    let svec = nalgebra::vector![1., 4., 2.];
    let dvec = nalgebra::dvector![1., 4., 2.];

    assert_eq!(Colx::<_>::into_owned(svec.clone()), svec);
    assert_eq!(Colx::<_>::clone_owned(&svec), svec);
    assert_eq!(Colx::<_>::max(&svec), Some(4.));
    assert_eq!(Colx::<_>::dim(&svec), Some(3));

    assert_eq!(Colx::<_>::into_owned(dvec.clone()), dvec);
    assert_eq!(Colx::<_>::clone_owned(&dvec), dvec);
    assert_eq!(Colx::<_>::max(&dvec), Some(4.));
    assert_eq!(Colx::<_>::dim(&dvec), Some(3));
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
