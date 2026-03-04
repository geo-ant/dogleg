use dogleg::{Dogleg, LeastSquaresProblem};
use nalgebra::{Matrix2, Vector2};

struct SimpleProblem {
    params: Vector2<f64>,
}

impl SimpleProblem {
    fn new(params: Vector2<f64>) -> Self {
        Self { params }
    }
}

impl LeastSquaresProblem<f64> for SimpleProblem {
    type Residuals = Vector2<f64>;
    type Parameters = Vector2<f64>;
    type Jacobian = Matrix2<f64>;

    fn set_params(&mut self, x: Self::Parameters) {
        self.params = x;
    }

    fn params(&self) -> Self::Parameters {
        self.params
    }

    fn residuals(&self) -> Option<Self::Residuals> {
        let x1 = self.params[0];
        let x2 = self.params[1];

        Some(nalgebra::vector![
            x1 * x2 - 1.,
            (x1 - 1.).powi(2) + (x2 - 1.).powi(2)
        ])
    }

    fn jacobian(&self) -> Option<Self::Jacobian> {
        let x1 = self.params[0];
        let x2 = self.params[1];

        Some(nalgebra::matrix![
            x2, x1;
            2.*(x1 - 1.) , 2.*(x2 - 1.)
        ])
    }
}

fn main() {
    // the initial parameters must be set
    // when constructing the problem.
    let problem = SimpleProblem::new(nalgebra::vector![2., 4.]);
    // construct a dogleg instance
    // and perform the minimization
    let (problem, _report) = Dogleg::new().minimize(problem).unwrap();
    println!("Expected minimum: (1,1)");
    println!("Found ({},{})", problem.params[0], problem.params[1]);
}
