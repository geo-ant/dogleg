use crate::{utility::enorm, Error};
use dogleg_matx::{Addx, Colx, Dotx, OwnedColx, Scalex};
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dim, IsContiguous, Matrix, OMatrix, OVector,
    RawStorage, RealField, Scalar, Storage, Vector,
};
use num_traits::{float::TotalOrder, ConstOne, Float};

/// helper trait to calculate ||J*v|| in an abstract fashion, where J is the
/// (scaled) jacobian matrix for the problem and v is a vector of suitable size.
/// Implementors of this trait typically store J directly or some decomposition
/// of J.
pub trait JacMatMulVecNorm<T, C>
where
    C: Dim,
    T: Scalar,
{
    /// calculate ||A*v||^2 using the given vector v and the internally stored
    /// information about A.
    fn jac_mul_vec_enorm<S>(&self, v: &Vector<T, C, S>) -> T
    where
        S: Storage<T, C> + IsContiguous;
}

impl<T, R, C, S> JacMatMulVecNorm<T, C> for Matrix<T, R, C, S>
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
    fn jac_mul_vec_enorm<S2>(&self, v: &Vector<T, C, S2>) -> T
    where
        S2: Storage<T, C> + IsContiguous,
    {
        let mv = self * v;
        enorm(&mv)
    }
}

/// The components for constructing the dogleg path for the next iteration,
/// as well as some additional information that is used in the following
/// step.
/// We use the Notation of Nocedal & Wright (2nd) ed, pp 66 - 76.
///
/// We also let the solver already
pub enum DoglegComponents<T, C, MVN>
where
    T: Scalar,
    C: Dim,
    DefaultAllocator: Allocator<C>,
    MVN: JacMatMulVecNorm<T, C>,
{
    /// Rndicates that the gtol criterium was satisfied, which means the iteration
    /// was finished successfully. The contained value is the actual value
    /// that satisfied gtol.
    GtolSatisfied(T),
    /// the solver has already decided that the first segment p_u of the
    /// dogleg path lies inside the trust region
    FirstSegmentInside {
        /// p_u
        p_u: OVector<T, C>,
        /// ||p_u|| <= delta
        pu_norm: T,
        /// a state cache that the solver implementation can use to store
        /// arbitrary information. This thing must also be able to calculate
        /// the result of ||J v||, where J is the jacobian of the residuals and
        /// v is a suitably sized vector.
        cached: MVN,
    },
    PathComponents {
        /// vector p_u for forming the first part of the dogleg segment
        p_u: OVector<T, C>,
        /// the p_b vector used to form the second part (p_b - p_u) of the
        /// dogleg path.
        p_b: OVector<T, C>,
        /// a state cache that the solver implementation can use to store
        /// arbitrary information. This thing must also be able to calculate
        /// the result of ||J v||, where J is the jacobian of the residuals and
        /// v is a suitably sized vector.
        cached: MVN,
    },
}

/// abstracts over the solution to the trust-region subproblem of finding the
/// optimal dogleg step given the current trust region radius.
/// This is returned by a dogled solver.
pub trait DoglegStepSolution<T, C>
where
    T: Scalar,
    C: Dim,
    DefaultAllocator: Allocator<C>,
{
    /// the solver can use this for internal caching and store whatever information
    /// it needs here. Importantly, this needs to be able to calculate ||J*v||,
    /// where J is the (scaled) Jacobian used to calculate the dogleg step
    /// and v is a suitably sized vector.
    type Cache: JacMatMulVecNorm<T, C>;
    fn unpack(self) -> (DoglegStepOLD<T, C>, Self::Cache);
}

/// this is the solution to the dogleg steps as determined by the dogleg step
/// solver, where p is the optimal next dogleg
pub struct DoglegStepOLD<T, C>
where
    T: Scalar,
    C: Dim,
    DefaultAllocator: Allocator<C>,
{
    /// next dogleg step, meaning x_{k+1} = x_k + p is the new iterate
    p: OVector<T, C>,
    /// the euclidean norm of ||p||, where it must be guaranteed that ||p|| <= delta,
    /// where delta is the trust region radius belonging to this step.
    p_norm: T,
}

pub enum DoglegSolverInput<MMN, VM, VN, Cache> {
    Init {
        // (scaled) Jacobian (matrix of size MxN)
        jacobian: MMN,
        // (not scaled) residuals (column vector with M elements)
        residuals: VM,
        // (scaled) gradient (column vector with N elements)
        gradient: VN,
    },
    Cached(Cache),
}

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
//@note(geo-ant) why do these generic have those weird names?
// MMN: means Matrix of Size MxN
// VM: column vector with M elements
// VN: column vector with N elemens
pub trait DoglegStepSolver<T, MMN, VM, VN> {
    /// this cache allows the dogleg solver to cache some calculations to be
    /// passed in at the next iteration which belongs to the same Jacobian
    /// and residual as has already been calculated.
    type Cache;

    /// the responsibility of this method is to calculate the dogleg components
    /// from the given inputs.
    /// See Nocedal and Wright, pp. 73 - 74 (for the dogleg part) and
    /// p. 246 for important notes that are particular for least squares, namely
    /// g = J^T r and B = J^T J (approx.), which togehter with the formulas
    /// on pp. 73 give the following:
    ///
    /// p_u is calculated as
    ///
    ///              ||g||^2
    /// p_u = -1 * -----------    g = u * g, where u: scalar, g: vector
    ///             ||J g||^2
    ///
    /// where g is the gradient of f : g = J^T r
    ///
    /// and p_b is the solution of min ||J p_b - (-r)||^2, where it
    /// makes sense to use some matrix decomposition rather than using the
    /// normal equations p_b = (J^T J)^-1 J^T r.
    ///
    /// We also return the matrix J or (if it makes sense) its decomposition
    /// so that we can use it to calculate ||J v||^2 for suitably sized vectors
    /// v in the downstream code. These things can be stored in the cache
    /// associated with this instance.
    fn dogleg_components<S1>(
        state: DoglegSolverInput<MMN, VM, VN, Self::Cache>,
        delta: T,
    ) -> Result<(DoglegStep<T, VN>, Self::Cache), Error>;
}

#[deprecated]
/// abstracts part of the algorithm whose responsibility it is to calculate
/// the dogleg components.
pub trait DoglegStepSolverOLD<T, R, C>
where
    C: Dim,
    T: Scalar,
    R: Dim,
    DefaultAllocator: nalgebra::allocator::Allocator<R>,
    DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    DefaultAllocator: nalgebra::allocator::Allocator<C>,
{
    /// type of the instance that allows us to calculate
    /// ||J*v|| for suitably sized vectors v, where J is the
    /// Jacobian of the residuals. Depending on the implementation, this
    /// thing could be the Jacobian itself or a matrix decomposition, which
    /// allows us to calculate the norm more efficiently.
    type Cache: JacMatMulVecNorm<T, C>;

    /// the responsibility of this method is to calculate the dogleg components
    /// from the given inputs.
    /// See Nocedal and Wright, pp. 73 - 74 (for the dogleg part) and
    /// p. 246 for important notes that are particular for least squares, namely
    /// g = J^T r and B = J^T J (approx.), which togehter with the formulas
    /// on pp. 73 give the following:
    ///
    /// p_u is calculated as
    ///
    ///              ||g||^2
    /// p_u = -1 * -----------    g = u * g, where u: scalar, g: vector
    ///             ||J g||^2
    ///
    /// where g is the gradient of f : g = J^T r
    ///
    /// and p_b is the solution of min ||J p_b - (-r)||^2, where it
    /// makes sense to use some matrix decomposition rather than using the
    /// normal equations p_b = (J^T J)^-1 J^T r.
    ///
    /// We also return the matrix J or (if it makes sense) its decomposition
    /// so that we can use it to calculate ||J v||^2 for suitably sized vectors
    /// v in the downstream code.
    fn dogleg_components<S1>(
        jacobian: OMatrix<T, R, C>,
        residuals: &Vector<T, R, S1>,
        delta: T,
    ) -> Result<DoglegComponents<T, C, Self::Cache>, Error>
    where
        S1: Storage<T, R>;
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
//@note(geo) we can also make this for different types PB, PU, in which case
// we have to use into_ownedx() for the return types and Option<PU::Ownedx> and
// constrain the PU : Colx<T, Ownedx= PB::Ownedx>. But I won't do it unless I
// have to.
pub fn dogleg_step<T, P>(pu: &P, pb: &P, delta: T) -> Option<P::Owned>
where
    T: Float + ConstOne,
    P: Colx<T, Owned = P> + Addx<T, P> + Dotx<T, P> + Scalex<T>,
    P::Owned: Scalex<T> + Addx<T, P> + Colx<T>,
{
    let pu_norm = pu.enorm();
    let pb_norm = pb.enorm();

    // we have to treat 3 cases differently:
    if pb_norm <= delta {
        // 1) in this case the entire dogleg lies inside the trust region radius
        // and we can just return the value for tau = 2, which is p_b
        Some(pb.clone_owned())
    } else if pu_norm >= delta {
        // 2) in this case the first part of the dogleg path lies inside
        // the trust region, so we can just find the tau in [0,1] for
        // which ||p|| = delta, which is just tau = delta/pu_norm.
        Some(pu.clone_owned().scale(delta / pu_norm))
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
        // pb - pu
        let pb_pu = pb.clone_owned().scaled_add(-T::ONE, &pu)?;
        let b = pu.dot(&pb_pu)?;

        let c = Float::powi(pb_pu.enorm(), 2);
        let d = Float::powi(delta, 2);
        let b_c = b / c;

        // this just checks for division by zero above which can't happen
        // mathematically, but numerically c can be small. The case c-> 0 implies
        // b/c -> inf, such that tau-1 = 0.
        if !b_c.is_finite() || b_c >= Float::sqrt(<T as Float>::max_value()) {
            return Some(pu.clone_owned());
        }
        let tau_minus_one = -b_c + Float::sqrt((d - a) / c + Float::powi(b_c, 2));
        pu.clone_owned()
            .scaled_add(T::ONE, &pb_pu.scale(tau_minus_one))
    }
}
