__version__ = '2.0.0.dev0'


import copy
import logging
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError, FitFailedWarning
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state

import mulearn.fuzzifier as fuzzifier
import mulearn.kernel as kernel
from mulearn.optimization import GurobiSolver

logger = logging.getLogger(__name__)

# relative tolerance (w.r.t. c) used to detect the bounds of the dual
# variables, in order to identify the unbounded support vectors
SV_RTOL = 1e-6

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)


class FuzzyInductor(RegressorMixin, BaseEstimator):
    """FuzzyInductor class."""

    def __init__(self,
                 c=1,
                 k=kernel.GaussianKernel(),
                 fuzzifier=fuzzifier.ExponentialFuzzifier(),
                 solver=GurobiSolver(),
                 random_state=None):
        r"""Create an instance of :class:`FuzzyInductor`.

        :param c: Trade-off constant, defaults to 1.
        :type c: `float`
        :param k: Kernel function, defaults to :class:`GaussianKernel()`.
        :type k: :class:`mulearn.kernel.Kernel`
        :param fuzzifier: fuzzifier mapping distance values to membership
           degrees, defaults to `ExponentialFuzzifier()`.
        :type fuzzifier: :class:`mulearn.fuzzifier.Fuzzifier`
        :param solver: Solver to be used to obtain the optimization problem
          solution, defaults to `GurobiSolver()`.
        :type solver: :class:`mulearn.optimization.Solver`
        :param random_state: Seed of the pseudorandom generator.
        :type random_state: `int`
        """
                     
        self.c = c
        self.k = k
        self.fuzzifier = fuzzifier
        self.solver = solver
        self.random_state = random_state

    def __repr__(self, **kwargs):
        return f"FuzzyInductor(c={self.c}, k={self.k}, f={self.fuzzifier}, " \
               f"solver={self.solver})"
    
    def __eq__(self, other):
        """Check equality w.r.t. other objects."""
        if type(self) is not type(other):
            return False
        equal = (self.c == other.c and self.k == other.k and
                 self.fuzzifier == other.fuzzifier)
        if hasattr(self, 'chis_') != hasattr(other, 'chis_'):
            return False
        if hasattr(self, 'chis_'):
            return equal and np.array_equal(self.chis_, other.chis_)
        return equal

    def x_to_sq_dist(self, X_new):
        X_new = np.array(X_new)
        t1 = self.k.compute(X_new, X_new)
        t2 = np.array([self.k.compute(x_i, X_new)
                        for x_i in self.X_]).transpose().dot(self.chis_)
        ret = t1 -2 * t2 + self.fixed_term_
        return ret

    def fit(self, X, y, warm_start=False):
        r"""Induce the membership function starting from a labeled sample.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the
          same length
        :param y: Membership for the vectors in `X`.
        :type y: iterable of `float` having the same length of `X`
        :param warm_start: flag triggering the non reinitialization of
          independent variables of the optimization problem, defaults to
          None.
        :type warm_start: `bool`
        :raises: ValueError if the values in `y` are not between 0 and 1, if
          `X` and have different lengths, or if `X` contains elements of
          different lengths.
        :returns: self -- the trained model.
        """

        X = check_array(X)
        X, y = check_X_y(X, y)

        for e in y:
            if e < 0 or e > 1:
                raise ValueError("`y` values should belong to [0, 1]")

        # tolerance for rounding errors when c = 1 / sum(y)
        if self.c * np.sum(y) < 1 - 1e-10:
            raise ValueError("the learning problem is infeasible, as "
                             f"c * sum(y) = {self.c * np.sum(y)} < 1: set c "
                             f"to at least {1 / np.sum(y)}")
            
        self.random_state = check_random_state(self.random_state)

        self.X_ = X

        if warm_start:
            check_is_fitted(self, ["chis_"])
            self.solver.initial_values = self.chis_

        if type(self.k) is kernel.PrecomputedKernel:
            idx = X.flatten()
            self.gram_ = self.k.kernel_computations[idx][:, idx]
        else:
            self.gram_ = np.array([self.k.compute(x1, X) for x1 in X])
        
        self.chis_ = self.solver.solve(X, y, self.c, self.gram_)

        # numerical solvers return values only approximately equal to the
        # bounds (e.g., 1e-12 instead of 0), hence the tolerance
        tol = SV_RTOL * self.c
        chi_SV_index = [i for i, (chi, mu) in enumerate(zip(self.chis_, y))
                        if -self.c * (1 - mu) + tol < chi < self.c * mu - tol]
        
        self.fixed_term_ = np.array(self.chis_).dot(self.gram_.dot(self.chis_))
        sq_dist = self.x_to_sq_dist(X)

        if chi_SV_index:
            sq_radius = np.mean(sq_dist[chi_SV_index])
        else:
            logger.warning('No support vectors found, squared radius '
                           'obtained through the pinball loss')
            sq_radius = self._pinball_sq_radius(sq_dist, y)

        self.fuzzifier.sq_radius_05 = sq_radius
        self.fuzzifier.fit(sq_dist, y, sq_radius)

        return self

    def _pinball_sq_radius(self, sq_dist, y):
        r"""Compute the squared radius when there are no support vectors.

        The squared radius minimizes the convex, piecewise linear function
        $r \mapsto r + c \sum_i \ell_{\mu_i}(R^2(x_i) - r)$, where $\ell_\mu$
        is the pinball loss, hence its minimum is attained at one of the
        squared distances of the training points."""

        def objective(r):
            t = sq_dist - r
            return r + self.c * np.sum(y * np.maximum(t, 0) +
                                       (1 - y) * np.maximum(-t, 0))

        return min(sq_dist, key=objective)

    def decision_function(self, X):
        r"""Compute predictions for the membership function.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the
          same length
        :returns: array of float -- the predictions for each value in `X`.
        """
        check_is_fitted(self, ['chis_'])
        X = check_array(X)
        return self.fuzzifier.get_membership(self.x_to_sq_dist(X))

    def predict(self, X, alpha=None):
        r"""Compute predictions for membership to the set.

        Predictions are either computed through the membership function (when
        `alpha` is set to a float in [0, 1]) or obtained via an $\alpha$-cut on
        the same function (when `alpha` is set to `None`).

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param alpha: $\alpha$-cut value, defaults to `None`.
        :type alpha: float
        :raises: ValueError if `alpha` is set to a value different from
          `None` and not included in $[0, 1]$.
        :returns: array of int -- the predictions for each value in `X`.
        """
        check_is_fitted(self, ['chis_'])
        X = check_array(X)
        mus = self.decision_function(X)
        if alpha is None:
            return mus
        else:
            if alpha < 0 or alpha > 1:
                raise ValueError("alpha cut value should belong to [0, 1]"
                                 f" (provided {alpha})")
            return np.array([1 if mu >= alpha else 0 for mu in mus])

    def score(self, X, y, **kwargs):
        r"""Compute the fuzzifier score.

        Score is obtained as the opposite of MSE between predicted
        membership values and labels.

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :param y: Labels containing the *gold standard* membership values
          for the vectors in `X`.
        :type y: iterable of `float` having the same length of `X`
        :returns: `float` -- opposite of MSE between the predictions for the
          elements in `X` w.r.t. the labels in `y`.
        """
        X, y = check_X_y(X, y)
        return -np.mean((self.decision_function(X) - y) ** 2)

    def get_profile(self):
      check_is_fitted(self, ['chis_'])
      return self.fuzzifier.get_profile(self.x_to_sq_dist(self.X_))

