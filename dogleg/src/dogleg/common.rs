use crate::{dogleg::report::TerminationFailure, MagicConst};
use dogleg_matx::{Addx, Colx, Dotx, Matx, MaxAbsx, OwnedColx, Scalex};
use num_traits::Float;

#[cfg(feature = "assert2")]
use assert2::debug_assert;

#[derive(Debug, Clone, PartialEq)]
//@note(geo-ant) why do these generic have those weird names?
// MMN: means Matrix of Size MxN
// VM: column vector with M elements
// VN: column vector with N elemens
pub struct DoglegStep<T, VN> {
    /// optimal next step `p` to take in this iteration
    pub p: VN,
    /// euclidean norm of this step
    pub p_norm: T,
    /// predicted reduction `m(0) - m(p)` of the model function
    /// if the step `p` was to be taken.
    pub predicted_reduction: T,
}

/// abstracts part of the algorithm whose responsibility it is to calculate
/// the dogleg components.
pub trait DoglegStepSolver<T>: Sized {
    /// type of the Jacobian for the least squares problem (matrix-type)
    type Jacobian: Matx<T>;
    /// type of the gradient for the problem (vector-type)
    type Gradient: OwnedColx<T>;
    /// type of the residuals (vector type)
    type Residuals: Colx<T>;
    /// Construct a dogleg solver to initialize an internal state using the
    /// Jacobian, residuals, and the gradient at the current parameters.
    fn init(
        jacobian: Self::Jacobian,
        residuals: Self::Residuals,
        gradient: Self::Gradient,
    ) -> Result<Self, TerminationFailure>;

    /// the responsibility of this method is to calculate the dogleg components
    /// from the given inputs at the current parameters. The Jacobian,
    /// residuals, and gradient that were last set using `init`
    /// will be used for this evaluation.
    ///
    /// See Nocedal and Wright, pp. 73 - 74 (for the dogleg part) and
    /// p. 246 for important notes that are particular for least squares, namely
    /// g = J^T r and B = J^T J (approx.), which togehter with the formulas
    /// on pp. 73 give the following:
    ///
    /// p_u is calculated as
    ///
    /// ```math
    ///              ||g||^2
    /// p_u = -1 * ----------- g  := u * g, where u: scalar, g: vector
    ///             ||J g||^2
    /// ```
    /// where g is the gradient of f, where g = J^T r
    ///
    /// and p_b is the solution of min ||J p_b - (-r)||^2, where it
    /// makes sense to use some matrix decomposition rather than using the
    /// normal equations p_b = (J^T J)^-1 J^T r.
    ///
    /// We also return the matrix J or (if it makes sense) its decomposition
    /// so that we can use it to calculate ||J v||^2 for suitably sized vectors
    /// v in the downstream code. These things can be stored in the cache
    /// associated with this instance.
    ///
    /// Returns the step and the next iteration of the internal solver state
    /// on success. An error otherwise. Takes self by values and returns self
    /// rather than &mut self because I like the by-value state pattern more.
    fn update_step(
        self,
        delta: T,
    ) -> Result<(DoglegStep<T, Self::Gradient>, Self), TerminationFailure>;
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
///
/// So this calculates
///          j_i^T r                 g_i
/// max_i  ----------------  = ---------------
///          ||j_i|| ||r||      ||j_i|| ||r||
///
/// where j_i (vec) is the i-th column of the jacobian and r (vec) is the
/// residual vector. g_i (scalar) is the i-th element of the gradient,
/// since g = J^T r.
pub(crate) fn minpack_gmax_calc<T, VN1, VN2>(
    jacobian_norms: &VN1,
    gradient: &VN2,
    residual_norm: T,
) -> Option<T>
where
    VN1: Colx<T>,
    VN2: MaxAbsx<T, VN1> + Colx<T>,
{
    assert_eq!(
        jacobian_norms.dim(),
        gradient.dim(),
        "jacobian must have same number of columns as gradient"
    );
    gradient.max_abs_scaled_div_elem(residual_norm, jacobian_norms)
}

/// this calculates the dogleg step from the component vectors p_b, p_u,
/// and the current trust region radius delta.
///
/// The dogleg path is parametrized using a real numer tau in [0,2]
///
/// ```math
///          { tau* p_u                    ; tau in [0,1]
/// p(tau) = {
///          { p_u + (tau-1) * (p_b - p_u) ; in (1,2]
/// ```
///
/// We return the largest step for which p(tau) <= detla
//@note(geo) we can also make this for different types PB, PU, in which case
// we have to use into_ownedx() for the return types and Option<PU::Ownedx> and
// constrain the PU : Colx<T, Ownedx= PB::Ownedx>. But I won't do it unless I
// have to.
pub fn traditional_dogleg_step<T, P>(
    pu: &P,
    pb: &P,
    delta: T,
) -> Result<P::Owned, TerminationFailure>
where
    T: Float + MagicConst + std::fmt::Debug,
    P: Colx<T, Owned = P> + Addx<T, P> + Dotx<T, P> + Scalex<T>,
    P::Owned: Scalex<T> + Addx<T, P> + Colx<T>,
{
    let pu_norm = pu.enorm();
    let pb_norm = pb.enorm();

    // we have to treat 3 cases differently:
    if pb_norm <= delta {
        // println!("Dogleg: choosing GN step.");
        // 1) in this case the entire dogleg lies inside the trust region radius
        // and we can just return the value for tau = 2, which is p_b
        Ok(pb.clone_owned())
    } else if pu_norm >= delta {
        // println!("Dogleg: choosing Cauchy step.");
        // 2) in this case the first part of the dogleg path lies inside
        // the trust region, so we can just find the tau in [0,1] for
        // which ||p|| = delta, which is just tau = delta/pu_norm.
        Ok(pu.clone_owned().scale(delta / pu_norm))
    } else {
        // println!("Dogleg: choosing interpolated step.");
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
        // pb - pu
        let pb_pu = pb
            .clone_owned()
            .scaled_add(-T::ONE, pu)
            .ok_or(TerminationFailure::WrongDimensions("dogleg step"))?;
        let b = pu
            .dot(&pb_pu)
            .ok_or(TerminationFailure::WrongDimensions("dogleg step"))?;

        let c = Float::powi(pb_pu.enorm(), 2);
        let d = Float::powi(delta, 2);
        let b_c = b / c;

        // this just checks for division by zero above which can't happen
        // mathematically, but numerically c can be small. The case c-> 0 implies
        // b/c -> inf, such that tau-1 = 0.
        if !b_c.is_finite() || b_c >= Float::sqrt(<T as Float>::max_value()) {
            return Ok(pu.clone_owned().scale(delta / pu_norm));
        }
        let tau_minus_one = -b_c + Float::sqrt((d - a) / c + Float::powi(b_c, 2));
        debug_assert!(tau_minus_one >= T::ZERO - T::EPSMCH);
        debug_assert!(tau_minus_one <= T::ONE + T::EPSMCH);

        // just a sanity check to see that I picked the correct formula for tau-1
        // (the square root allows for the +/- but my solution should only
        // be the correct one that is always in [0,1].
        let tau_minus_one_alt = -b_c - Float::sqrt((d - a) / c + Float::powi(b_c, 2));
        debug_assert!(tau_minus_one_alt < T::ZERO || tau_minus_one_alt > T::ONE);

        let p = pu
            .clone_owned()
            .add(&pb_pu.scale(tau_minus_one))
            .ok_or(TerminationFailure::WrongDimensions("dogleg step"))?;
        let p_norm = p.enorm();

        if p_norm > delta {
            Ok(p.scale(delta / p_norm))
        } else {
            Ok(p)
        }

        // some sanity checks with some generous bounds for numerical problems
        // debug_assert!(Float::powi(p_norm, 2) <= delta * delta + T::EPSMCH * T::ONE_HUNDRED);
        // debug_assert!(p_norm >= pu_norm - T::EPSMCH * T::TEN);
        // debug_assert!(p_norm <= pb_norm + T::EPSMCH * T::TEN);
        // @todo(geo) maybe add more logic to restrict the p step to the feasible
        // range.
        // Ok(p)
    }
}
