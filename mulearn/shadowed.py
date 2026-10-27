"""Induction of shadowed sets.

A shadowed set is learned by extending the optimization problem of
:class:`mulearn.FuzzyInductor` so as to also find two squared distances
`theta_1 <= theta_0` from the center of the learned sphere, delimiting core,
shadow and exclusion. Two formulations are available:

* 'original' learns a linear fuzzifier decreasing from 1 to 0 as the squared
  distance goes from `theta_1` to `theta_0`, under the constraint that the
  images of all training points lie within distance `theta_0` from the
  center, and then cuts the corresponding fuzzy set at a level balancing the
  fuzziness of core, shadow and exclusion;
* 'three-way' only penalizes the points with positive membership lying in the
  exclusion (beyond `theta_0`) and those with membership below 1 lying in the
  core (within `theta_1`), points in the shadow costing nothing, while the
  width `theta_0 - theta_1` of the shadow is penalized.
"""

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import mulearn.kernel as kernel
from mulearn.optimization import nearest_psd

try:
    import clarabel
except ImportError:  # pragma: no cover
    clarabel = None

# relative tolerance (w.r.t. c) used to detect the bounds of the dual
# variables, in order to identify the unbounded support vectors
SV_RTOL = 1e-6

# optimal cut of the linear fuzzifier, balancing the fuzziness of core and
# exclusion against that of the shadow
OPTIMAL_ALPHA = np.sqrt(2) / 4

# value returned by `predict` for the points in the shadow
SHADOW = 0.5

FORMULATIONS = ('original', 'three-way')


class ShadowedSetInductor(BaseEstimator):
    r"""Shadowed set inductor.

    With the 'original' formulation the learning problem is

    .. math::
      \min \frac{\theta_1 + \theta_0}{2} + C \sum_i (\xi_i + \tau_i)
           + D (\theta_0 - \theta_1)

    subject to $\mu_i \|\Phi(x_i) - a\|^2 \leq \mu_i (\theta_1 + \theta_0)/2
    + \xi_i$, $(1 - \mu_i) \|\Phi(x_i) - a\|^2 \geq (1 - \mu_i)
    (\theta_1 + \theta_0)/2 - \tau_i$, $\|\Phi(x_i) - a\|^2 \leq \theta_0$,
    $\theta_1 \leq \theta_0$ and $\xi_i, \tau_i \geq 0$, whose dual reads

    .. math::
      \min_{\chi, \beta} (\chi + \beta)^\top K (\chi + \beta)
                         - \mathrm{diag}(K)^\top (\chi + \beta)

    subject to $\sum_i (\chi_i + \beta_i) = 1$, $\sum_i \beta_i \leq 2D$,
    $-C(1 - \mu_i) \leq \chi_i \leq C \mu_i$ and $\beta_i \geq 0$. The center
    of the sphere is $a = \sum_i (\chi_i + \beta_i) \Phi(x_i)$, and its squared
    radius $(\theta_1 + \theta_0)/2$ is the squared distance from $a$ of the
    images of the points such that $-C(1 - \mu_i) < \chi_i < C \mu_i$;
    moreover, $\theta_0$ is the largest between the squared radius and the
    squared distances from $a$ of the training points. Core and exclusion
    are obtained by cutting the linear fuzzifier at levels $1 - \alpha$ and
    $\alpha$.

    With the 'three-way' formulation the learning problem is

    .. math::
      \min \theta_0 + D (\theta_0 - \theta_1) + C \sum_i (\xi_i + \tau_i)

    subject to $\mu_i \|\Phi(x_i) - a\|^2 \leq \mu_i \theta_0 + \xi_i$,
    $(1 - \mu_i) \|\Phi(x_i) - a\|^2 \geq (1 - \mu_i) \theta_1 - \tau_i$,
    $\theta_1 \leq \theta_0$ and $\xi_i, \tau_i \geq 0$, whose dual reads

    .. math::
      \min_{p, q} (p - q)^\top K (p - q) - \mathrm{diag}(K)^\top (p - q)

    subject to $\sum_i (p_i - q_i) = 1$, $\sum_i q_i \leq D$,
    $0 \leq p_i \leq C \mu_i$ and $0 \leq q_i \leq C (1 - \mu_i)$. The
    center of the sphere is $a = \sum_i (p_i - q_i) \Phi(x_i)$, and core and
    exclusion contain the points whose images lie, respectively, within
    squared distance $\theta_1$ and beyond squared distance $\theta_0$ from
    $a$. When $D \geq C \sum_i (1 - \mu_i)$ the problem coincides with that
    of :class:`mulearn.FuzzyInductor`, and $\theta_1 = \theta_0$.
    """

    def __init__(self, c=1, d=0.1, k=kernel.GaussianKernel(),
                 formulation='original', alpha=OPTIMAL_ALPHA, tol=1e-8,
                 max_iter=200):
        """Create an instance of :class:`ShadowedSetInductor`.

        :param c: Trade-off constant $C$ for the slack variables, defaults
          to 1.
        :type c: `float`
        :param d: Trade-off constant $D$ for the width of the shadow,
          defaults to 0.1.
        :type d: `float`
        :param k: Kernel function, defaults to :class:`GaussianKernel()`.
        :type k: :class:`mulearn.kernel.Kernel`
        :param formulation: either 'original' or 'three-way' (see the class
          documentation), defaults to 'original'.
        :type formulation: `str`
        :param alpha: level of the cut defining core ($\\mu \\geq 1 - \\alpha$),
          shadow and exclusion ($\\mu \\leq \\alpha$) in the 'original'
          formulation, defaults to $\\sqrt 2 / 4$, which equally balances the
          fuzziness of the shadow and that of core and exclusion.
        :type alpha: `float`
        :param tol: Tolerance of the Clarabel solver, defaults to 1e-8.
        :type tol: `float`
        :param max_iter: Maximum number of iterations of the Clarabel solver,
          defaults to 200.
        :type max_iter: `int`
        """
        self.c = c
        self.d = d
        self.k = k
        self.formulation = formulation
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

    def _solve_qp(self, P, q, A, b, n_equalities):
        """Minimize x^T P x / 2 + q^T x subject to A x + s = b, the first
        `n_equalities` components of s being null and the remaining ones
        nonnegative."""
        if clarabel is None:
            raise ValueError('clarabel not available')
        cones = [clarabel.ZeroConeT(n_equalities),
                 clarabel.NonnegativeConeT(A.shape[0] - n_equalities)]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = self.max_iter
        settings.tol_gap_abs = settings.tol_gap_rel = self.tol
        settings.tol_feas = self.tol
        solution = clarabel.DefaultSolver(sparse.csc_matrix(np.triu(P)), q,
                                          A.tocsc(), b, cones,
                                          settings).solve()
        if solution.status not in (clarabel.SolverStatus.Solved,
                                   clarabel.SolverStatus.AlmostSolved):
            raise RuntimeError(f'clarabel: {solution.status}')
        return np.array(solution.x)

    def _solve(self, gram, mu):
        """Solve the dual problem, returning the two vectors of variables
        ((chi, beta) or (p, q), depending on the formulation)."""
        n = len(mu)
        gram, is_psd = nearest_psd(gram)
        if not is_psd:
            raise ValueError('the Gram matrix is not positive semidefinite')
        eye, zero = sparse.eye(n), sparse.csc_matrix((n, n))
        ones, nothing = (sparse.csc_matrix(np.ones((1, n))),
                         sparse.csc_matrix((1, n)))

        if self.formulation == 'original':
            # the objective only depends on chi + beta; constraints: sum of
            # chi + beta, sum of beta, bounds of chi, nonnegativity of beta
            sign = 1
            A = sparse.vstack([sparse.hstack([ones, ones]),
                               sparse.hstack([nothing, ones]),
                               sparse.hstack([eye, zero]),
                               sparse.hstack([-eye, zero]),
                               sparse.hstack([zero, -eye])])
            b = np.concatenate([[1, 2 * self.d], self.c * mu,
                                self.c * (1 - mu), np.zeros(n)])
        else:
            # the objective only depends on p - q; constraints: sum of p - q,
            # sum of q, bounds of p and of q
            sign = -1
            A = sparse.vstack([sparse.hstack([ones, -ones]),
                               sparse.hstack([nothing, ones]),
                               sparse.hstack([eye, zero]),
                               sparse.hstack([-eye, zero]),
                               sparse.hstack([zero, eye]),
                               sparse.hstack([zero, -eye])])
            b = np.concatenate([[1, self.d], self.c * mu, np.zeros(n),
                                self.c * (1 - mu), np.zeros(n)])
        P = 2 * np.block([[gram, sign * gram], [sign * gram, gram]])
        q = -np.concatenate([np.diag(gram), sign * np.diag(gram)])
        x = self._solve_qp(P, q, A, b, 1)
        return x[:n], x[n:]

    def fit(self, X, y):
        """Learn the shadowed set from data.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: Membership degrees in [0, 1] of the vectors in `X`.
        :type y: iterable of `float` having the same length of `X`
        :raises: ValueError if the formulation is unknown or the problem is
          infeasible, that is, if $C \\sum_i \\mu_i < 1 - 2D$ ('original') or
          $C \\sum_i \\mu_i < 1$ ('three-way').
        :returns: self
        """
        if self.formulation not in FORMULATIONS:
            raise ValueError(f'unknown formulation {self.formulation!r}')
        X, y = check_X_y(X, y, dtype=float)
        bound = 1 - 2 * self.d if self.formulation == 'original' else 1
        if self.c * np.sum(y) < bound - 1e-10:
            raise ValueError('the learning problem is infeasible: c * sum(y) '
                             f'must be at least {bound:g}')

        self.X_ = X
        self.gram_ = np.array([self.k.compute(x, X) for x in X])
        first, second = self._solve(self.gram_, y)
        if self.formulation == 'original':
            self.chis_, self.betas_ = first, second
            self.weights_ = first + second
        else:
            self.p_, self.q_ = first, second
            self.weights_ = first - second
        self.fixed_term_ = self.weights_ @ self.gram_ @ self.weights_
        sq_dist = self.x_to_sq_dist(X)

        if self.formulation == 'original':
            tol = SV_RTOL * self.c
            free = ((self.chis_ > -self.c * (1 - y) + tol) &
                    (self.chis_ < self.c * y - tol))
            sq_radius = (np.mean(sq_dist[free]) if free.any()
                         else self._primal_sq_radius(sq_dist, y))
            self.theta_0_ = float(max(sq_dist.max(), sq_radius))
            self.theta_1_ = float(2 * sq_radius - self.theta_0_)
        else:
            self.theta_1_, self.theta_0_ = self._primal_thresholds(sq_dist, y)
        self.sq_radius_ = (self.theta_1_ + self.theta_0_) / 2
        return self

    def _primal_sq_radius(self, sq_dist, y):
        """Squared radius minimizing the primal objective of the 'original'
        formulation for the learned center, used when no unbounded support
        vector is found: as theta_0 equals max(max(sq_dist), radius), the
        objective is piecewise linear in the radius, and its minimum is
        attained at some sq_dist[i]."""
        top = sq_dist.max()

        def objective(r):
            t = sq_dist - r
            return (r + 2 * self.d * (max(top, r) - r) + self.c *
                    np.sum(y * np.maximum(t, 0) + (1 - y) * np.maximum(-t, 0)))

        return min(sq_dist, key=objective)

    def _primal_thresholds(self, sq_dist, y):
        """theta_1 and theta_0 minimizing the primal objective of the
        'three-way' formulation for the learned center. The objective is
        piecewise linear and separable in the two thresholds, besides the
        constraint theta_1 <= theta_0, so that its minimum is attained at some
        sq_dist[i]; among the optimal solutions, the one having the narrowest
        shadow is returned."""
        candidates = np.unique(sq_dist)
        positive = self.c * np.array([np.sum(y * np.maximum(sq_dist - t, 0))
                                      for t in candidates])
        negative = self.c * np.array([np.sum((1 - y) *
                                             np.maximum(t - sq_dist, 0))
                                      for t in candidates])
        cost_0 = (1 + self.d) * candidates + positive
        cost_1 = -self.d * candidates + negative
        scale = 1e-9 * max(1, np.abs(candidates).max())
        theta_0 = candidates[np.flatnonzero(cost_0 <= cost_0.min() + scale)[0]]
        theta_1 = candidates[np.flatnonzero(cost_1 <= cost_1.min() + scale)[-1]]
        if theta_1 <= theta_0:
            return float(theta_1), float(theta_0)
        # the constraint is active: a single threshold
        joint = cost_0 + cost_1
        theta = candidates[np.argmin(joint)]
        return float(theta), float(theta)

    def x_to_sq_dist(self, X):
        """Squared distances of the images of `X` from the center."""
        check_is_fitted(self, ['weights_'])
        X = check_array(X, dtype=float)
        cross = np.array([self.k.compute(x_i, X) for x_i in self.X_]).T
        return (self.k.compute(X, X) - 2 * cross @ self.weights_ +
                self.fixed_term_)

    def fuzzy_membership(self, X):
        """Membership to the learned fuzzy set, linearly decreasing from 1 to
        0 as the squared distance goes from theta_1 to theta_0."""
        sq_dist = self.x_to_sq_dist(X)
        if self.theta_0_ == self.theta_1_:
            return (sq_dist <= self.theta_0_).astype(float)
        return np.clip((self.theta_0_ - sq_dist) /
                       (self.theta_0_ - self.theta_1_), 0, 1)

    def predict(self, X):
        """Membership to the learned shadowed set: 1 in the core, 0 in the
        exclusion and `SHADOW` (that is, 0.5) in the shadow, where membership
        is unknown."""
        if self.formulation == 'original':
            mu = self.fuzzy_membership(X)
            return np.where(mu >= 1 - self.alpha, 1.0,
                            np.where(mu <= self.alpha, 0.0, SHADOW))
        sq_dist = self.x_to_sq_dist(X)
        return np.where(sq_dist <= self.theta_1_, 1.0,
                        np.where(sq_dist >= self.theta_0_, 0.0, SHADOW))
