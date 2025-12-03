use crate::{col_assert_relative_eq, mat_assert_relative_eq};

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
#[should_panic]
fn col_assert_relative_eq_test_should_panic3() {
    col_assert_relative_eq!(faer::col!(1., 2.), faer::col!(1.));
}

#[test]
fn col_assert_relative_eq_test_works() {
    col_assert_relative_eq!(faer::col!(1., 2.), faer::col!(1., 2.), epsilon = 1e-10);
    col_assert_relative_eq!(faer::col!(1., 2.), faer::col!(1.1, 1.9), epsilon = 0.2);
}

#[test]
#[should_panic]
fn mat_assert_relative_eq_test_should_panic() {
    mat_assert_relative_eq!(
        faer::mat!([1., 2.], [3., 4.]),
        faer::mat!([1.000000001, 2.], [3., 4.])
    );
}

#[test]
#[should_panic]
fn mat_assert_relative_eq_test_should_panic2() {
    mat_assert_relative_eq!(
        faer::mat!([1., 2.], [3., 4.]),
        faer::mat!([1., 2.000000001], [3., 4.])
    );
}

#[test]
#[should_panic]
fn mat_assert_relative_eq_test_should_panic3() {
    mat_assert_relative_eq!(faer::mat!([1., 2.], [3., 4.]), faer::mat!([1., 2.]));
}

#[test]
fn mat_assert_relative_eq_test_works() {
    mat_assert_relative_eq!(
        faer::mat!([1., 2.], [3., 4.]),
        faer::mat!([1., 2.], [3., 4.]),
        epsilon = 1e-10
    );
    mat_assert_relative_eq!(
        faer::mat!([1., 2.], [3., 4.]),
        faer::mat!([1.1, 1.9], [3., 4.]),
        epsilon = 0.2
    );
}
