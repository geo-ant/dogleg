//! This code is based on `test_examples_gen.rs` from the `levenberg-marquardt` crate.
//!
//! The original license for this code is as such:
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

use crate::ceres_solve_with_dogleg;
use levenberg_marquardt::LeastSquaresProblem;
use levmar_problems::{assert_fp_eq, problems::*, utils::differentiate_numerically};
use nalgebra::*;

#[test]
fn test_linear_full_rank() {
    let mut problem = LinearFullRank::new(OVector::<f64, U5>::zeros(), 10);
    let initial = OVector::<f64, U5>::from_column_slice(&[1., 1., 1., 1., 1.]);

    // check derivative implementation
    problem.set_params(&OVector::<f64, U5>::from_column_slice(&[
        0.5488135039273248,
        0.7151893663724195,
        0.6027633760716439,
        0.5448831829968969,
        0.4236547993389047,
    ]));
    let jac_num = differentiate_numerically(&mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_fp_eq!(jac_num, jac_trait, epsilon = 1e-5);

    problem.set_params(&initial.clone());

    let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        report.objective_function,
        2.5000000000000004,
        epsilon = 1e-6
    );
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U5>::from_column_slice(&[
            -1.,
            -1.0000000000000004,
            -1.,
            -1.0000000000000004,
            -1.
        ]),
        epsilon = 1e-6
    );
    // assert_fp_eq!(report.objective_function, 2.5000000000000004);

    let mut problem = LinearFullRank::new(OVector::<f64, U5>::zeros(), 50);
    let initial = OVector::<f64, U5>::from_column_slice(&[1., 1., 1., 1., 1.]);

    problem.set_params(&initial.clone());
    let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        report.objective_function,
        22.500000000000004,
        epsilon = 1e-6
    );
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U5>::from_column_slice(&[
            -0.9999999999999953,
            -1.0000000000000049,
            -0.9999999999999976,
            -0.9999999999999956,
            -0.9999999999999991
        ]),
        epsilon = 1e-6
    );
}

#[test]
// see MGH paper: https://www.cmor-faculty.rice.edu/~yzhang/caam454/nls/MGH.pdf
// problem 33
fn test_linear_rank1() {
    let mut problem = LinearRank1::new(OVector::<f64, U5>::zeros(), 10);
    let initial = OVector::<f64, U5>::from_column_slice(&[1., 1., 1., 1., 1.]);

    // check derivative implementation
    problem.set_params(&OVector::<f64, U5>::from_column_slice(&[
        0.6458941130666561,
        0.4375872112626925,
        0.8917730007820798,
        0.9636627605010293,
        0.3834415188257777,
    ]));
    let jac_num = differentiate_numerically(&mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_fp_eq!(jac_num, jac_trait, epsilon = 1e-5);

    problem.set_params(&initial.clone());
    let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    let m = problem.m as f64;
    // NOTE: expected minimum as given in the MGH paper must be scaled by 0.5
    // to compare to our objective function.
    let fmin_mgh = m * (m - 1.) / (2. * (2. * m + 1.));

    let sum_jxj: f64 = problem
        .params
        .iter()
        .enumerate()
        .map(|(j, xj)| xj * ((j + 1) as f64))
        .sum();
    // every point where \sum_j {j*x_j} = 3/(2m+1) is a valid minimum
    assert_fp_eq!(sum_jxj, 3. / (2. * m + 1.), epsilon = 1e-6);
    assert_fp_eq!(report.objective_function, 0.5 * fmin_mgh, epsilon = 1e-6);

    let mut problem = LinearRank1::new(OVector::<f64, U5>::zeros(), 50);
    let initial = OVector::<f64, U5>::from_column_slice(&[1., 1., 1., 1., 1.]);

    problem.set_params(&initial.clone());
    let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    let m = problem.m as f64;
    // NOTE: expected minimum as given in the MGH paper must be scaled by 0.5
    // to compare to our objective function.
    let fmin_mgh = m * (m - 1.) / (2. * (2. * m + 1.));

    let sum_jxj: f64 = problem
        .params
        .iter()
        .enumerate()
        .map(|(j, xj)| xj * ((j + 1) as f64))
        .sum();
    // every point where \sum_j {j*x_j} = 3/(2m+1) is a valid minimum
    assert_fp_eq!(sum_jxj, 3. / (2. * m + 1.), epsilon = 1e-6);
    assert_fp_eq!(report.objective_function, 0.5 * fmin_mgh, epsilon = 1e-6);
}

// NOTE(geo-ant) see above
#[test]
fn test_linear_rank1_zero_columns() {
    todo!()
}

#[test]
fn test_rosenbrock() {
    let mut problem = Rosenbruck {
        params: OVector::<f64, U2>::zeros(),
    };
    let initial = OVector::<f64, U2>::from_column_slice(&[-1.2, 1.]);

    // check derivative implementation
    problem.set_params(&OVector::<f64, U2>::from_column_slice(&[
        0.08712929970154071,
        0.02021839744032572,
    ]));
    let jac_num = differentiate_numerically(&mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_fp_eq!(jac_num, jac_trait, epsilon = 1e-5);

    problem.set_params(&initial.clone());
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(report.objective_function, 0.0, epsilon = 1e-6);
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U2>::from_column_slice(&[1., 1.]),
        epsilon = 1e-6
    );
    problem.set_params(&initial.map(|x| 10. * x));
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(report.objective_function, 0.0, epsilon = 1e-6);
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U2>::from_column_slice(&[1., 1.]),
        epsilon = 1e-6
    );
    problem.set_params(&initial.map(|x| x * 100.));
    let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(report.objective_function, 0.0, epsilon = 1e-6);
    assert_fp_eq!(report.objective_function, 0.0);
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U2>::from_column_slice(&[1., 1.]),
        epsilon = 1e-6
    );
}

#[test]
// see https://rdrr.io/github/jlmelville/funconstrain/man/helical.html
// minimum should be f = 0 at (1,0,0)
fn test_helical_valley() {
    let mut problem = HelicalValley {
        params: OVector::<f64, U3>::zeros(),
    };
    let initial = OVector::<f64, U3>::from_column_slice(&[-1., 0., 0.]);

    // check derivative implementation
    problem.set_params(&OVector::<f64, U3>::from_column_slice(&[
        0.832619845547938,
        0.7781567509498505,
        0.8700121482468192,
    ]));
    let jac_num = differentiate_numerically(&mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_fp_eq!(jac_num, jac_trait, epsilon = 1e-5);

    problem.set_params(&initial.clone());
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U3>::from_column_slice(&[1., -6.243301596789443e-18, 0.]),
        epsilon = 1e-6
    );
    assert_fp_eq!(report.objective_function, 0.0, epsilon = 1e-6);

    problem.set_params(&initial.map(|x| x * 10.));
    let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U3>::from_column_slice(&[1., 6.563910805155555e-21, 0.]),
        epsilon = 1e-6
    );
    assert_fp_eq!(report.objective_function, 0.0, epsilon = 1e-6);

    // NOTE(geo-ant) this actually fails
    // problem.set_params(&initial.map(|x| x * 100.));
    // let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    // assert_fp_eq!(
    //     problem.params,
    //     OVector::<f64, U3>::from_column_slice(&[1., -1.9721522630525295e-30, 0.]),
    //     epsilon = 1e-6
    // );
}

#[test]
fn test_powell_singular() {
    let mut problem = PowellSingular {
        params: OVector::<f64, U4>::zeros(),
    };
    let initial = OVector::<f64, U4>::from_column_slice(&[3., -1., 0., 1.]);

    // check derivative implementation
    problem.set_params(&OVector::<f64, U4>::from_column_slice(&[
        0.978618342232764,
        0.7991585642167236,
        0.46147936225293185,
        0.7805291762864555,
    ]));
    let jac_num = differentiate_numerically(&mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_fp_eq!(jac_num, jac_trait, epsilon = 1e-5);

    problem.set_params(&initial.clone());
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(report.objective_function, 0.0, epsilon = 1e-6);
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U4>::from_column_slice(&[
            1.6521175961683935e-17,
            -1.6521175961683934e-18,
            2.6433881538694683e-18,
            2.6433881538694683e-18
        ]),
        epsilon = 1e-3
    );

    problem.set_params(&initial.map(|x| x * 10.));
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(report.objective_function, 0.0, epsilon = 1e-6);
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U4>::from_column_slice(&[
            2.0167451125102287e-20,
            -2.0167451125102287e-21,
            3.2267921800163004e-21,
            3.2267921800163004e-21
        ]),
        epsilon = 1e-3
    );

    problem.set_params(&initial.map(|x| x * 100.));
    let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(report.objective_function, 0.0, epsilon = 1e-6);
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U4>::from_column_slice(&[
            3.2267921800163781e-18,
            -3.2267921800163780e-19,
            5.1628674880262125e-19,
            5.1628674880262125e-19
        ]),
        epsilon = 1e-3
    );
}

#[test]
// this is actually a failed minimization because this is a local minimum,
// NOT the global minimum. The global minimum should be at (5,4),
// see https://rdrr.io/github/jlmelville/funconstrain/man/freud_roth.html.
// The package is pretty useful overall https://github.com/jlmelville/funconstrain
// because it gives us a test suite of problems for unconstrained optimization,
// which has a high overlap with this one here.
fn test_freudenstein_roth() {
    let mut problem = FreudensteinRoth {
        params: OVector::<f64, U2>::zeros(),
    };
    let initial = OVector::<f64, U2>::from_column_slice(&[0.5, -2.]);

    // check derivative implementation
    problem.set_params(&OVector::<f64, U2>::from_column_slice(&[
        0.11827442586893322,
        0.6399210213275238,
    ]));
    let jac_num = differentiate_numerically(&mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_fp_eq!(jac_num, jac_trait, epsilon = 1e-5);

    problem.set_params(&initial.clone());
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U2>::from_column_slice(&[11.412484465499368, -0.8968279137315035]),
        epsilon = 1e-3
    );
    // see the MGH paper: https://www.cmor-faculty.rice.edu/~yzhang/caam454/nls/MGH.pdf
    // problem (2)
    assert_fp_eq!(report.objective_function, 0.5 * 48.9842, epsilon = 1e-3);

    problem.set_params(&initial.map(|x| x * 10.));
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U2>::from_column_slice(&[11.413004661474561, -0.8967960386859591]),
        epsilon = 1e-3
    );
    assert_fp_eq!(report.objective_function, 0.5 * 48.9842, epsilon = 1e-3);

    problem.set_params(&initial.map(|x| x * 100.));
    let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U2>::from_column_slice(&[11.412781785788198, -0.8968051074920677]),
        epsilon = 1e-2
    );
    assert_fp_eq!(report.objective_function, 0.5 * 48.9842, epsilon = 1e-3);
}

#[test]
fn test_bard() {
    let mut problem = Bard {
        params: OVector::<f64, U3>::zeros(),
    };
    let initial = OVector::<f64, U3>::from_column_slice(&[1., 1., 1.]);

    // check derivative implementation
    problem.set_params(&OVector::<f64, U3>::from_column_slice(&[
        0.1433532874090464,
        0.9446689170495839,
        0.5218483217500717,
    ]));
    let jac_num = differentiate_numerically(&mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_fp_eq!(jac_num, jac_trait, epsilon = 1e-5);

    problem.set_params(&initial.clone());
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U3>::from_column_slice(&[
            0.0824105765758334,
            1.1330366534715044,
            2.343694638941154
        ]),
        epsilon = 1e-3
    );
    // see MGH paper: https://www.cmor-faculty.rice.edu/~yzhang/caam454/nls/MGH.pdf
    // see problem (8), the first optimum where no position of the minimum is
    // given, but we'll just take what we had in levenberg-marquardt
    assert_fp_eq!(report.objective_function, 0.5 * 8.21487e-3, epsilon = 1e-8);

    // NOTE(geo-ant) those problems fail with the levmar crate as well.
    // see https://rdrr.io/github/jlmelville/funconstrain/man/bard.html where it says:
    //  > Minima: f = 8.214877e-3 at c(0.08241056, 1.133036, 2.343695)
    //  > Solvers terminate with f near 17 for parameter 1 in 0.84 to 0.89
    //  > approximately and large negative values of the other two parameters.

    // NOTE(geo-ant) this fails!
    // problem.set_params(&initial.map(|x| x * 10.));
    // let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    // assert_fp_eq!(problem.params[0], 8.40e-01, epsilon = 1e-3);
    // assert2::assert!(problem.params[1] <= -1e+05);
    // assert2::assert!(problem.params[2] <= -1e+04);

    // NOTE(geo-ant) interestingly, this passes although the starting point
    // is farther away.
    problem.set_params(&initial.map(|x| x * 100.));
    let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(report.objective_function, 0.5 * 17.4286, epsilon = 1e-4);
    assert_fp_eq!(problem.params[0], 8.40e-01, epsilon = 1e-3);
    assert2::assert!(problem.params[1] <= -1e+05);
    assert2::assert!(problem.params[2] <= -1e+04);
}

#[test]
// see https://rdrr.io/github/jlmelville/funconstrain/man/kow_osb.html
// Minima: f = 3.07505...e-4; and f = 1.02734...e-3 at (Inf, -14.07..., -Inf, -Inf).
// so we can't reasonably expect the same minima to be produced as in levmar
// (although sometimes it seems they are).
// I also have to think about the values of the objective function, since
// we also can't reasonably expect them to match, though I think my objective
// function is 1/2* the given objective function, this seems to match pretty well.
fn test_kowalik_osborne() {
    let mut problem = KowalikOsborne {
        params: OVector::<f64, U4>::zeros(),
    };
    let initial = OVector::<f64, U4>::from_column_slice(&[0.25, 0.39, 0.415, 0.39]);

    // check derivative implementation
    problem.set_params(&OVector::<f64, U4>::from_column_slice(&[
        0.4146619399905236,
        0.26455561210462697,
        0.7742336894342167,
        0.45615033221654855,
    ]));
    let jac_num = differentiate_numerically(&mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_fp_eq!(jac_num, jac_trait, epsilon = 1e-5);

    problem.set_params(&initial.clone());
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        report.objective_function,
        0.00015375280229088455,
        epsilon = 1e-5
    );
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U4>::from_column_slice(&[
            0.19280781047624931,
            0.1912626533540709,
            0.12305280104693087,
            0.13605322115051674
        ]),
        epsilon = 1e-3
    );

    problem.set_params(&initial.map(|x| x * 10.));
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        report.objective_function,
        0.000513671535424324,
        epsilon = 1e-6
    );
    // this is actually not a very good solution, but this is the best that
    // the ceres dogleg is able to do.
    assert2::assert!(problem.params[0] > 1e2);
    assert_fp_eq!(problem.params[1], -1.4075880312939264e+01, epsilon = 1e-1);
    assert2::assert!(problem.params[2] < -1e3);
    assert2::assert!(problem.params[3] < -1e3);

    // this actually finds a different (but worse) minimum than levenberg marquardt,
    // but a valid local minimum nonetheless. Finds the same minimum as with
    // the starting point above
    problem.set_params(&initial.map(|x| x * 100.));
    let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        report.objective_function,
        0.000513671535424324,
        epsilon = 1e-6
    );
    assert2::assert!(problem.params[0] > 1e2);
    assert_fp_eq!(problem.params[1], -1.4075880312939264e+01, epsilon = 1e-1);
    assert2::assert!(problem.params[2] < -1e3);
    assert2::assert!(problem.params[3] < -1e3);
}

// see https://rdrr.io/github/jlmelville/funconstrain/man/meyer.html
// >  Minima: the MGH (1981) only provides the optimal value, with
// > f = 87.9458.... Meyer and Roth (1972) give the optimal parameter values
// > as (0.0056, 6181.4, 345.2), with f = 88.
#[test]
fn test_meyer() {
    let mut problem = Meyer {
        params: OVector::<f64, U3>::zeros(),
    };
    let initial = OVector::<f64, U3>::from_column_slice(&[2.0e-02, 4.0e+03, 2.5e+02]);

    // check derivative implementation
    problem.set_params(&OVector::<f64, U3>::from_column_slice(&[
        0.5684339488686485,
        0.018789800436355142,
        0.6176354970758771,
    ]));
    let jac_num = differentiate_numerically(&mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_fp_eq!(jac_num, jac_trait, epsilon = 1e-5);

    problem.set_params(&initial.clone());
    let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();

    // #[cfg(feature = "minpack-compat")]
    assert_fp_eq!(
        report.objective_function,
        43.972927585355414,
        epsilon = 1e-5
    );
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U3>::from_column_slice(&[
            5.6096364710271603e-03,
            6.1813463462865056e+03,
            3.4522363462414097e+02
        ]),
        epsilon = 1e-2
    );
    // NOTE(geo-ant) this is a real failure case in levmar as well, no need
    // to actuall test that for compatibility with minpack
    // problem.set_params(&initial.map(|x| x * 10.));
    // #[allow(unused_variables)]
    // let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    // assert_fp_eq!(report.objective_function, 324272.94195590157);
    // assert_fp_eq!(
    //     problem.params,
    //     OVector::<f64, U3>::from_column_slice(&[
    //         6.825607045203072e-12,
    //         3.514599603833739e+04,
    //         9.220431522058431e+02
    //     ])
    // );
}

#[test]
// see https://rdrr.io/github/jlmelville/funconstrain/man/watson.html
fn test_watson() {
    let mut problem = Watson::new(OVector::<f64, U6>::zeros(), 6);
    let initial = OVector::<f64, U6>::from_column_slice(&[0., 0., 0., 0., 0., 0.]);

    // check derivative implementation
    problem.set_params(&OVector::<f64, U6>::from_column_slice(&[
        0.6120957227224214,
        0.6169339968747569,
        0.9437480785146242,
        0.6818202991034834,
        0.359507900573786,
        0.43703195379934145,
    ]));
    let jac_num = differentiate_numerically(&mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_fp_eq!(jac_num, jac_trait, epsilon = 1e-5);

    problem.set_params(&initial.clone());
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        report.objective_function,
        0.001143835026786261,
        epsilon = 1e-5
    );
    // see https://github.com/jlmelville/funconstrain/blob/master/R/20_watson.R
    // (look for xmin = c(...)). These are actually the parameters at the reported
    // minimum.
    //
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U6>::from_column_slice(&[
            -0.01572496150837828,
            1.0124348823296545,
            -0.23299172238767143,
            1.260431011028177,
            -1.5137303139441967,
            0.9929972729184159
        ]),
        epsilon = 1e-3
    );
    problem.set_params(&initial.map(|x| x + 10.));
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        report.objective_function,
        0.0011438350267831846,
        epsilon = 1e-5
    );
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U6>::from_column_slice(&[
            -0.015725190138667525,
            1.0124348586010505,
            -0.23299154584382673,
            1.2604293208916204,
            -1.5137277670657403,
            0.9929957342632777
        ]),
        epsilon = 1e-3
    );
    problem.set_params(&initial.map(|x| x + 100.));
    let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        report.objective_function,
        0.0011438350268716062,
        epsilon = 1e-5
    );
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U6>::from_column_slice(&[
            -0.01572470197125869,
            1.0124349092565827,
            -0.2329919227616415,
            1.2604329292955434,
            -1.513733204527061,
            0.9929990192232175
        ]),
        epsilon = 1e-3
    );

    let mut problem = Watson::new(OVector::<f64, U9>::zeros(), 9);
    let initial = OVector::<f64, U9>::from_column_slice(&[0., 0., 0., 0., 0., 0., 0., 0., 0.]);

    problem.set_params(&initial.clone());
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        report.objective_function,
        6.998800690506343e-07,
        epsilon = 1e-5
    );
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U9>::from_column_slice(&[
            -1.5307064416628804e-05,
            9.9978970393459676e-01,
            1.4763963491099890e-02,
            1.4634233014597900e-01,
            1.0008210945482034,
            -2.6177311207051202,
            4.1044031394335869,
            -3.1436122623624456,
            1.052626403787601
        ]),
        epsilon = 1e-2
    );
    problem.set_params(&initial.map(|x| x + 10.));
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        report.objective_function,
        6.998800690471173e-07,
        epsilon = 1e-5
    );
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U9>::from_column_slice(&[
            -1.5307036495997912e-05,
            9.9978970393194666e-01,
            1.4763963693703627e-02,
            1.4634232829808710e-01,
            1.0008211030105516,
            -2.6177311405327139,
            4.1044031644962153,
            -3.1436122785677023,
            1.052626408013118
        ]),
        epsilon = 1e-2
    );
    problem.set_params(&initial.map(|x| x + 100.));
    let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        report.objective_function,
        6.998800690486009e-07,
        epsilon = 1e-5
    );
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U9>::from_column_slice(&[
            -1.5306952335212645e-05,
            9.9978970395837152e-01,
            1.4763962518529752e-02,
            1.4634234109641628e-01,
            1.0008210472912598,
            -2.6177310157356275,
            4.1044030142719174,
            -3.1436121860244794,
            1.0526263851676092
        ]),
        epsilon = 1e-2
    );

    let mut problem = Watson::new(OVector::<f64, U12>::zeros(), 12);
    let initial =
        OVector::<f64, U12>::from_column_slice(&[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]);

    problem.set_params(&initial.clone());
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    // this is from the MGH paper
    assert_fp_eq!(report.objective_function, 0.5 * 4.72238e-10, epsilon = 1e-5);
    // this is from the original levenberg-marquardt crate. The minimum is the
    // same but the parameters are not and I can't find a source that tells
    // me what the correct parameters at the minimum should be.
    // assert_fp_eq!(
    //     report.objective_function,
    //     2.3611905506971735e-10,
    //     epsilon = 1e-5
    // );
    // assert_fp_eq!(
    //     problem.params,
    //     OVector::<f64, U12>::from_column_slice(&[
    //         -6.6380604677589803e-09,
    //         1.0000016441178612,
    //         -5.6393221015137217e-04,
    //         3.4782054049969546e-01,
    //         -1.5673150405406330e-01,
    //         1.05281517698587,
    //         -3.2472711527607245,
    //         7.2884348965512684,
    //         -1.0271848239579612e+01,
    //         9.0741136457303284,
    //         -4.5413754661102059,
    //         1.0120118884445952
    //     ]),
    //     epsilon = 1e-2
    // );
    problem.set_params(&initial.map(|x| x + 10.));
    let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    let _ = problem;
    assert_fp_eq!(
        report.objective_function,
        2.361190552167311e-10,
        epsilon = 1e-5
    );
    // same note as above, the objective function seems to be correct
    // assert_fp_eq!(
    //     problem.params,
    //     OVector::<f64, U12>::from_column_slice(&[
    //         -6.6380604668544608e-09,
    //         1.0000016441178616,
    //         -5.6393221029791976e-04,
    //         3.4782054050317829e-01,
    //         -1.5673150408911857e-01,
    //         1.0528151771767233,
    //         -3.2472711533826666,
    //         7.2884348978198767,
    //         -1.0271848241212496e+01,
    //         9.0741136470182528,
    //         -4.5413754666784278,
    //         1.0120118885519702
    //     ]),
    //     epsilon = 1e-2
    // );

    // NOTE: with these starting parameters, the CERES algorithm fails
    // problem.set_params(&initial.map(|x| x + 100.));
    // let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    // assert_fp_eq!(
    //     report.objective_function,
    //     2.361190551562772e-10,
    //     epsilon = 1e-5
    // );
    // assert_fp_eq!(
    //     problem.params,
    //     OVector::<f64, U12>::from_column_slice(&[
    //         -6.6380604636792693e-09,
    //         1.0000016441178616,
    //         -5.6393221027197340e-04,
    //         3.4782054050235750e-01,
    //         -1.5673150407932457e-01,
    //         1.0528151771168239,
    //         -3.2472711531707001,
    //         7.2884348973610109,
    //         -1.0271848240595697e+01,
    //         9.0741136465161336,
    //         -4.5413754664517798,
    //         1.0120118885084435
    //     ]),
    //     epsilon = 1e-2
    // );
}

#[test]
#[ignore = "ceres dogleg fails here"]
// see https://rdrr.io/github/jlmelville/funconstrain/man/beale.html
fn test_beale() {
    let mut problem = Beale {
        params: OVector::<f64, U2>::zeros(),
    };
    let initial = OVector::<f64, U2>::from_column_slice(&[2.5, 1.]);

    // check derivative implementation
    problem.set_params(&OVector::<f64, U2>::from_column_slice(&[
        0.6976311959272649,
        0.06022547162926983,
    ]));
    let jac_num = differentiate_numerically(&mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap();
    assert_fp_eq!(jac_num, jac_trait, epsilon = 1e-5);

    problem.set_params(&initial.clone());
    let (mut problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        report.objective_function,
        6.982085570779134e-07,
        epsilon = 1e-4
    );
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U2>::from_column_slice(&[2.8252463853580405, 0.4595596246635109])
    );
    problem.set_params(&initial.map(|x| x - 0.5));
    let (problem, report) = ceres_solve_with_dogleg(problem).unwrap();
    assert_fp_eq!(
        report.objective_function,
        5.355422879172696e-16,
        epsilon = 1e-4
    );
    assert_fp_eq!(
        problem.params,
        OVector::<f64, U2>::from_column_slice(&[2.9989956785046323, 0.4997826037201959])
    );
}
