use crate::utility::enorm;
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dim, IsContiguous, Matrix, OVector, RawStorage,
    RealField, Scalar, Storage, Vector,
};
use num_traits::{float::TotalOrder, ConstOne, Float};

/// helper trait to calculate ||A*v||^2 in an abstract fashion. Implementors
/// of this trait should store A or some decomposition of A and then the associated
/// function can be used to calculate ||A*v||^2, where A is a matrix, v is a
/// suitably sized vector and ||.||^2 is the squared euclidean norm.
pub(crate) trait MatMulVecNorm<T, D>
where
    D: Dim,
    T: RealField,
{
    /// calculate ||A*v||^2 using the given vector v and the internally stored
    /// information about A.
    fn mul_vec_enorm<S>(&self, v: &Vector<T, D, S>) -> T
    where
        S: Storage<T, D> + IsContiguous;
}

impl<T, R, C, S> MatMulVecNorm<T, C> for Matrix<T, R, C, S>
where
    T: RealField + Float,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    DefaultAllocator: nalgebra::allocator::Allocator<R>,
    DefaultAllocator: nalgebra::allocator::Allocator<C>,
    DefaultAllocator: nalgebra::allocator::Allocator<R, Const<1>>,
{
    #[inline]
    fn mul_vec_enorm<S2>(&self, v: &Vector<T, C, S2>) -> T
    where
        S2: Storage<T, C> + IsContiguous,
    {
        let mv = self * v;
        enorm(&mv)
    }
}

pub(crate) enum DoglegComponents<T, R, C, MM>
where
    T: Scalar + RealField,
    R: Dim,
    C: Dim,
    MM: MatMulVecNorm<T, R>,
    DefaultAllocator: Allocator<R> + Allocator<C>,
{
    /// Rndicates that the gtol criterium was satisfied, which means the iteration
    /// was finished successfully. The contained value is the actual value
    /// that satisfied gtol.
    GtolSatisfied(T),
    /// The components for constructing the dogleg path for the next iteration,
    /// as well as some additional information that is used in the following
    /// step.
    /// We use the Notation of Nocedal & Wright (2nd) ed, pp 66 - 76.
    Components {
        /// the gradient of the objective function f = 1/2 ||r(x)||^2. See below,
        /// this can be used to calculate the p_u component of the dogleg path.
        g: OVector<T, R>,
        /// a scalar u, where p_u = u*g, where p_u is the first dogleg path
        /// component. We don't return p_u directly because we might need
        /// g for further calculations.
        u: T,
        /// the p_b vector used to form the second part (p_b - p_u) of the
        /// dogleg path.
        p_b: OVector<T, C>,
        /// utility to calculate ||J*v||^2, which is used in downstream
        /// calculations, where J is the Jacobi matrix of the residuals.
        jacmul: MM,
    },
}

/// this performs the calculation which gives us the value to compare against
/// gtol. The name is a bit stupid, because this doesn't calculate gtol, but
/// I don't know how to name it better. This function takes the jacobian (J)
/// and the residuals (r) and outputs a single value. I'll explain what it calculates
/// below. If that value is <= gtol, then the gtol check passes and iteration
/// is finished successfully.
///
/// # What does the gtol check actually calculate?
///
/// The [minpack user guide](https://cds.cern.ch/record/126569/files/CM-P00068642.pdf)
/// does a great job of explaining the criterion on page 21,22. The idea is
/// that at the minimum the gradient J^T r should be the zero vector. This criterion
/// tests for this a little smarter than just comparing each element to zero.
/// What we do is realize that each component of (J^T r)_i is the multiplication
/// of the vectors (j_i^T r), where j_i is the i-th COLUMN of the jacobian.
/// That means at the minimum, the columns of the Jacobian j_i are orthogonal
/// to the residuals. That means cos(theta_i) = 0, where theta is the angle
/// between the vectors, so gval_i = cos(theta_i) = (j_i^T r)/(||j_i||*||r||).
/// What we return is gval_i which is a better criterion for checking against
/// the gradient being zero due to the normalization
pub fn gtol_calc<T, R, C, S, S2>(jacobian: &Matrix<T, R, C, S>, residuals: &Vector<T, R, S2>) -> T
where
    T: RealField + Float + TotalOrder,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C> + Storage<T, R, C>,
    S2: RawStorage<T, R> + Storage<T, R>,
{
    let rnorm = enorm(&residuals);

    jacobian
        .column_iter()
        .map(|j| {
            let jnorm = enorm(&j);
            let jtr = j.dot(residuals);

            jtr / (jnorm * rnorm)
        })
        .max_by(TotalOrder::total_cmp)
        .unwrap_or(Float::infinity())
}

/// this calculates the dogleg step from the component vectors p_b, p_u,
/// and the current trust region radius delta.
///
/// The dogleg path is parametrized using a real numer tau in [0,2]
///
///          { tau* p_u                    ; tau in [0,1]
/// p(tau) = {
///          { p_u + (tau-1) * (p_b - p_u) ; in (1,2]
///
/// We return the largest step for which p(tau) <= detla
pub fn dogleg_step<T, C, S1, S2>(p_u: OVector<T, C>, p_b: OVector<T, C>, delta: T) -> OVector<T, C>
where
    T: RealField + Float,
    C: Dim,
    // S1: RawStorage<T, C> + Storage<T, C>,
    // S2: RawStorage<T, C> + Storage<T, C>,
    DefaultAllocator: nalgebra::allocator::Allocator<C>,
{
    let pu_norm = enorm(&p_u);
    let pb_norm = enorm(&p_b);

    // we have to treat 3 cases differently:
    if pb_norm <= delta {
        // 1) in this case the entire dogleg lies inside the trust region radius
        // and we can just return the value for tau = 2, which is p_b
        p_b.clone_owned()
    } else if pu_norm >= delta {
        // 2) in this case the first part of the dogleg path lies inside
        // the trust region, so we can just find the tau in [0,1] for
        // which ||p|| = delta, which is just tau = delta/pu_norm.
        p_u * (delta / pu_norm)
    } else {
        // 3) in this case the rust region intersects somewhere inside the
        // second part of the dogleg and we have to do some algebra
        // to find the correct tau in [1,2].
        //
        // This boils down to solving the quadratic equation
        //
        // ||p_u + (tau-1) * (p_b - p_u)||^2 = delta^2
        //
        // => ||p_u||^2 + 2(tau-1) p_u^T (p_b-p_u) + (tau-1)^2 ||p_b - p_u||^2 = delta^2
        //
        // If we substitue x = tau-1, we see that this is a quadratic equation
        // in x and then with a little bit of rearranging, we can find a solution
        let a = Float::powi(pu_norm, 2);
        let pb_pu = &p_b - &p_u;
        let b = p_u.dot(&pb_pu);
        let c = Float::powi(enorm(&pb_pu), 2);
        let d = Float::powi(delta, 2);
        let b_c = b / c;

        // this just checks for division by zero above which can't happen
        // mathematically, but numerically c can be small. The case c-> 0 implies
        // b/c -> inf, such that tau-1 = 0.
        if !b_c.is_finite() || b_c >= Float::sqrt(<T as Float>::max_value()) {
            return p_u;
        }
        let tau_minus_one = -b_c + Float::sqrt((d - a) / c + Float::powi(b_c, 2));
        p_u + pb_pu * tau_minus_one
    }
}
