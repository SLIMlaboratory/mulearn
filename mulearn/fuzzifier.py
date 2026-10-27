
"""This module implements fuzzifiers used in mulearn.
"""

import copy
import logging
import warnings

import json_fix
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_X_y, check_array

np.seterr(over='ignore')

logger = logging.getLogger(__name__)

def check_is_fitted(fuzzifier, attributes):
    """Raise NotFittedError if `fuzzifier` lacks any of the `attributes`.

    Replaces sklearn's check_is_fitted, which since scikit-learn 1.8 only
    accepts estimators (fuzzifiers are not BaseEstimator subclasses)."""
    if isinstance(attributes, str):
        attributes = [attributes]
    if not all(hasattr(fuzzifier, a) for a in attributes):
        raise NotFittedError(f"This {type(fuzzifier).__name__} instance is "
                             "not fitted yet.")

def _safe_exp(r):
    with np.errstate(over="raise"):
        try:
            return np.exp(r)
        except FloatingPointError:
            return 1

def exp_clip(a):
    return np.where(a > 0, 1, np.exp(a))


def _core_guess(squared_R, mu, squared_radius, threshold):
    """Initial guess for the squared distance at which membership starts
    decreasing from 1: the median squared distance of the points whose
    membership is at least `threshold` times the maximal one, moved halfway
    towards the smallest squared distance (or slightly below
    `squared_radius`, if all points lie on the sphere) if it is not below
    `squared_radius`: this happens, e.g., when these points lie on the
    learnt sphere, and would make the fixed profiles degenerate."""
    squared_R = np.asarray(squared_R, dtype=float)
    mu = np.asarray(mu, dtype=float)
    guess = np.median(squared_R[mu >= max(mu) * threshold])
    if guess >= squared_radius:
        lowest = min(min(squared_R), squared_radius * (1 - 1e-3))
        guess = (lowest + squared_radius) / 2
    return guess


def _core_upper(squared_radius):
    """Upper bound for the core squared radius in the fixed profiles, which
    must be strictly smaller than the squared radius of the sphere (otherwise
    the profile is undefined)."""
    return squared_radius * (1 - 1e-9)


class Fuzzifier:
    """Base class for fuzzifiers.

    The base class for fuzzifiers is Fuzzifier: it exposes a basic constructor
    which is called from the subclasses, and two methods `get_membership`
    (returning the membership function inferred from data) and `get_profile`
    computing information exploitable in order to visualize the fuzzifier
    in graphical form."""

    def __init__(self, profile):
        self.profile = profile

    def _fit_as_fixed(self, squared_R, mu, squared_radius):
        """Fit the parameters of the `'fixed'` profile, to be used when
        fitting with the current profile fails. The `profile` attribute is
        left unchanged, so that the hyperparameters set by the user are
        preserved (e.g., when the fuzzifier is cloned in a grid search)."""

        logger.warning(f"Fitting with profile {self.profile!r} failed, "
                       "using the parameters of the 'fixed' profile")
        profile = self.profile
        self.profile = 'fixed'
        try:
            self.fit(squared_R, mu, squared_radius)
        finally:
            self.profile = profile

    def fit(self, squared_R, mu, squared_radius):
        """Fit the fuzzifier on training data.

        :param squared_R: iterable of squared distance of the images of vectors
          in data space w.r.t. the center of the fuzzy set in feature space.
        :type squared_R: iterable of `float`
        :param mu: membership degrees of the vectors having originated
          `squared_R`.
        :type mu: vector of floats having the same length of `squared_R`
        :param squared_radius: radius of the fuzzy set in feature space.
        :type squared_radius: float
        :returns: self -- the fitted fuzzifier.
        """
        raise NotImplementedError(
            'The base class does not implement the `fit` method')

    def get_membership(self, R_2):
        """Return the membership degrees for the given squared distances.

        :param R_2: squared distances from the center of the fuzzy set in
          feature space.
        :type R_2: iterable of `float`
        :raises: NotFittedError if `fit` has not been called
        :returns: array -- the corresponding membership degrees.
        """
        raise NotImplementedError(
            'The base class does not implement the `get_membership` method')

    def get_profile(self, squared_R):
        r"""Return information about the learnt membership function profile.

        The profile of a membership function $\mu: X \rightarrow [0, 1]$ is
        intended here as the associated function $p: \mathbb R^+ \rightarrow
        [0, 1]$ still returning membership degrees, but considering its
        arguments in the feature space. More precisely, if `X` contains the
        values $x_1, \dots, x_n$, $R^2$ is the function mapping any point in
        data space into the squared distance between its image and the center
        $a$ of the learnt fuzzy set in feature space, the function
        `get_profile` computes the following information about $p$:

        * a list $r_\mathrm{data} = [ R^2(x_i), i = 1, \dots, n]$ containing
          the distances between the images of the points in `X` and $a$;
        * a list $\tilde{r}_\mathrm{data}$ containing 200 possible
          distances between $a$ and the image of a point in data space, evenly
          distributed between $0$ and $\max r_{\mathrm{data}}$;
        * a list $e = [\hat\mu(r_i), r_i \in \tilde{r}_{\mathrm{data}}]$
          gathering the profile values for each element in
          $\tilde{r}_{\mathrm{data}}$.

        This information can be used in order to graphically show the
        membership profile, which is always plottable, whereas the membership
        function isn't mostly of the time (unless the elements in `X` are
        either one- or bidimensional vectors).

        :param X: Vectors in data space.
        :type X: iterable of `float` vectors having the same length
        :returns: list -- $[r_{\mathrm{data}}, \tilde{r}_\mathrm{data}, e]$.

        """
        rdata_synth = np.linspace(0, max(squared_R) * 1.1, 200)
        estimate = self.get_membership(rdata_synth)
        return [squared_R, rdata_synth, estimate]

    def __str__(self):
        """Return the string representation of a fuzzifier."""
        return self.__repr__()

    def __eq__(self, other):
        """Check fuzzifier equality w.r.t. other objects."""
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Check fuzzifier inequality w.r.t. other objects."""
        return not self == other

    def __hash__(self):
        """Generate hashcode for a fuzzifier."""
        return hash(self.__repr__())

    @staticmethod
    def __nonzero__():
        """Check if a fuzzifier is non-null."""
        return True
    
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __json__(self):
        return {'class': self.__class__.__name__, 'profile': self.profile}
    
    def __repr__(self):
        """Return the python representation of the fuzzifier."""
        arg = self.profile if self.profile != self.default_profile else ''
        return f'{self.__class__.__name__}({arg})'


class CrispFuzzifier(Fuzzifier):
    """Crisp fuzzifier.

    Fuzzifier corresponding to a crisp (classical) set: membership is always
    equal to either $0$ or $1$."""

    default_profile = "fixed"
    def __init__(self, profile=default_profile):
        r"""Create an instance of :class:`CrispFuzzifier`.

        :param profile: method to be used in order to build the fuzzifier
          profile: `'fixed'` relies on the radius of the sphere defining
          the fuzzy set core, while `'infer'` fits a generic threshold
          function on the provided examples.
        :type profile: str
        """
        super().__init__(profile)
    
    def fit(self, squared_R, mu, squared_radius):
        r"""Fit the fuzzifier on training data.

        :param squared_R: iterable of squared distance of the images of vectors
          in data space w.r.t. the center of the fuzzy set in feature space.
        :type squared_R: iterable of `float`
        :param mu: membership degrees of the vectors having originated
          `squared_R`.
        :type mu: vector of floats having the same length of `squared_R`
        :param squared_radius: radius of the fuzzy set in feature space.
        :type squared_radius: float

        :raises: ValueError if self.profile is not set either to `'fixed'` or
          to `'infer'`.

        The fitting process is done considering a threshold-based membership
        function, in turn corresponding to a threshold-based profile of the
        form

        .. math::
          p(r) = \begin{cases} 1 & \text{if $r \leq r_\text{crisp}$,} \\
                               0 & \text{otherwise.} \end{cases}

        The threshold $r_\text{crisp}$ is set to the learnt square radius of
        the sphere when the `profile` attribute of the class have been set to
        `'fixed'`. When it has been set to `'infer'`, the threshold minimizing
        the squared error on the provided examples is found exactly: as the
        error only changes when the threshold crosses one of the values in
        `squared_R`, these values are sorted and the threshold is set to the
        midpoint of the best interval they define.
        """

        self.name = 'Crisp'
        self.latex_name = '$\\hat\\mu_{\\text{crisp}}$'

        assert len(squared_R) == len(mu)

            
        if self.profile == "fixed":
            self.threshold_ = squared_radius

        elif self.profile == "infer":
            order = np.argsort(squared_R)
            r = np.asarray(squared_R, dtype=float)[order]
            m = np.asarray(mu, dtype=float)[order]

            # sse[k]: squared error when the k closest points get membership 1
            sse = np.concatenate(([0], np.cumsum((1 - m) ** 2))) + \
                  np.concatenate((np.cumsum((m ** 2)[::-1])[::-1], [0]))

            # the threshold cannot separate points at the same distance
            n = len(r)
            valid = np.ones(n + 1, dtype=bool)
            valid[1:n] = r[:-1] < r[1:]
            k = np.flatnonzero(valid)[np.argmin(sse[valid])]

            if k == 0:
                self.threshold_ = r[0] / 2 if r[0] > 0 else -1.0
            elif k == n:
                self.threshold_ = r[-1]
            else:
                self.threshold_ = (r[k - 1] + r[k]) / 2

        else:
            raise ValueError("'profile' parameter should either be equal to "
                             f"'fixed' or 'infer' (provided: {self.profile})")

        return self

    def get_membership(self, R_2):
        check_is_fitted(self, 'threshold_')
        return np.array([1 if r_2 <= self.threshold_ else 0 for r_2 in R_2])


class LinearFuzzifier(Fuzzifier):
    """Linear fuzzifier.

    Fuzzifier corresponding to a fuzzy set whose membership in feature space
    linearly decreases from 1 to 0."""

    default_profile = "fixed"
    def __init__(self, profile=default_profile):
        r"""Create an instance of :class:`LinearFuzzifier`.
        :param profile: method to be used in order to build the fuzzifier
          profile: `'fixed'` relies on the radius of the sphere defining
          the fuzzy set core, `'triangular'` aims at inducing a triangular or
          trapezoidal membership function in data space and `'infer'` fits a
          generic threshold function on the provided examples.
        :type profile: str"""
        super().__init__(profile)

    def fit(self, squared_R, mu, squared_radius):
        r"""Fit the fuzzifier on training data.

        :param squared_R: iterable of squared distance of the images of vectors
          in data space w.r.t. the center of the fuzzy set in feature space.
        :type squared_R: iterable of `float`
        :param mu: membership degrees of the vectors having originated
          `squared_R`.
        :type mu: vector of floats having the same length of `squared_R`
        :param squared_radius: radius of the fuzzy set in feature space.
        :type squared_radius: float

        :raises: ValueError if self.profile is not set to `'fixed'`,
          `'triangular`' or `'infer'`.

        The fitting process is done considering a membership function
        linearly decreasing from $1$ to $0$, in turn corresponding to a profile
        having the general form

        .. math::
          p(r) = \begin{cases} 1 & \text{if $r \leq r_1$,} \\
                               l(r) & \text{if $r_1 < r \leq r_0$,} \\
                               0 & \text{otherwise.} \end{cases}

        The free parameters are chosen in order to guarantee continuity;
        moreover, when the `profile` attribute of the class have been set to
        `'fixed'` the membership profile will be equal to 0.5 when $r$ is equal
        to the learnt square radius of the sphere, and induced via
        interpolation of `X` and `y` when it is has been set to `'infer'`.
        """

        self.name = 'Linear'
        self.latex_name = '$\\hat\\mu_{\\text{lin}}$'
        
        assert len(squared_R) == len(mu)
        
        r_2_1_guess = _core_guess(squared_R, mu, squared_radius, 0.99)
        r_2_0_guess = np.median([r2 for r2, m in zip (squared_R, mu)
                                              if m <= min(mu)*1.01])
        r_2_05_guess = (r_2_1_guess + r_2_0_guess) / 2

                                      
        if self.profile == 'fixed':
            def r2_to_mu(R_2, r_2_1):

                result = [np.clip(1 - 0.5 * (r_2-r_2_1)/(squared_radius-r_2_1),
                                  0, 1) for r_2 in R_2]
                
                return result

            upper = _core_upper(squared_radius)
            [r_2_1_opt], _ = curve_fit(r2_to_mu, squared_R, mu,
                                       p0=(min(max(r_2_1_guess, 0), upper),),
                                       bounds=((0,), (upper,)))
            self.slope_ = -1 / (2 * (squared_radius - r_2_1_opt))
            self.intercept_ = 1 + r_2_1_opt / (2 * (squared_radius - r_2_1_opt))
        
            
        elif self.profile == 'triangular':
            def r2_to_mu(R_2, r_2_05):
                # TODO: check
                # return [np.clip(1 - r_2 / (2 * squared_radius),
                #                 0, 1) - r_2_1
                #         for r_2 in R_2]
                return [np.clip(1 - r_2 / (2 * r_2_05),
                                0, 1)
                        for r_2 in R_2]

            try:
                [r_2_05_opt], _ = curve_fit(r2_to_mu, squared_R, mu,
                                            p0=(r_2_05_guess,),
                                            bounds=((0,), (np.inf,)))
                self.slope_ = -1 / (2 * r_2_05_opt)
                self.intercept_ = 1
            except RuntimeError:
                # interpolation could not take place, fall back to fixed profile
                self._fit_as_fixed(squared_R, mu, squared_radius)

        elif self.profile == 'infer':

            def r2_to_mu(R_2, r_2_1, r_2_0):
                return [np.clip(1 - (r_2 - r_2_1) / (r_2_0 - r_2_1), 0, 1)
                        for r_2 in R_2]

            try:
                p_opt, _ = curve_fit(r2_to_mu, squared_R, mu,
                                    p0=(r_2_1_guess, r_2_0_guess), 
                                    bounds=((-np.inf, -np.inf),
                                            (np.inf, np.inf,)))
                r_2_1_opt, r_2_0_opt = p_opt
                self.slope_ = -1 / (r_2_0_opt - r_2_1_opt)
                self.intercept_ = 1 + r_2_1_opt / (r_2_0_opt - r_2_1_opt)
            except RuntimeError:
                # interpolation could not take place, fall back to fixed profile
                self._fit_as_fixed(squared_R, mu, squared_radius)

        else:
            raise ValueError("'profile' parameter should be equal to "
                             "'fixed', 'infer' or 'triangular' "
                             f"(provided value: {self.profile})")
        if self.slope_ > 0:
            logger.warning('Profile fitting returned a positive slope '
                            f'({self.slope_})')
        if self.intercept_ < 0:
            logger.warning('Profile fitting returned a negative intercept '
                            f'({self.intercept_})')
        
        return self

    def get_membership(self, R_2):
        check_is_fitted(self, ['slope_', 'intercept_'])
        return np.array([np.clip(self.slope_ * r_2 + self.intercept_, 0, 1)
                         for r_2 in R_2])


class ExponentialFuzzifier(Fuzzifier):
    """Exponential fuzzifier.

    Fuzzifier corresponding to a fuzzy set whose membership in feature space
    exponentially decreases from 1 to 0."""

    default_profile = "fixed"
    def __init__(self, profile=default_profile):
        r"""Create an instance of :class:`ExponentialFuzzifier`.

        :param profile: method to be used in order to build the fuzzifier
          profile: `'fixed'` relies on the radius of the sphere defining
          the fuzzy set core, `'infer'` fits the profile function on the
          provided examples, and a numeric value allows for manually setting
          the exponential decay of the fuzzifier.
        :type profile: str or numeric value"""

        super().__init__(profile)

    def fit(self, squared_R, mu, squared_radius):
        r"""Fit the fuzzifier on training data.

        :param squared_R: iterable of squared distance of the images of vectors
          in data space w.r.t. the center of the fuzzy set in feature space.
        :type squared_R: iterable of `float`
        :param mu: membership degrees of the vectors having originated
          `squared_R`.
        :type mu: vector of floats having the same length of `squared_R`
        :param squared_radius: radius of the fuzzy set in feature space.
        :type squared_radius: float
        
        :raises: ValueError if self.profile is not set either to `'fixed'`,
          `'infer'` or to a numeric value in $[0, 1]$.

        In this fuzzifier, the function that transforms the square distance
        between the center of the learnt sphere and the image of a point in
        the original space into a membership degree has the form

        .. math::
          \mu(r) = \begin{cases}  1    & \text{if $r \leq r_1$,} \\
                                  e(r) & \text{otherwise,}
                   \end{cases}

        where $e$ is an exponential function decreasing from 1 to 0. The
        shape of this function is chosen so that:
        - when `self.profile`=`'fixed'` the membership profile will be equal
          equal to 0.5 when $r$ is equal to the learnt square radius of the
          sphere,
        - induced via interpolation of `squared_R` and `mu` when
          `self.profile`=`'infer'` and
        - manually set when `self.profile` is a number $\alpha \in [0, 1]$, the
          latter now implying that the fuzzifier value will be 0.5 exactly when
          its argument equals the $\alpha$-quantile of the squared distances
          of the provided examples in feature space.
        """

        self.name = "Exponential"
        self.latex_name = r"$\hat\mu_{\text{exp}}$"

        assert len(squared_R) == len(mu)

        if isinstance(self.profile, (int, float)):
            if self.profile < 0 or self.profile > 1:
                raise ValueError("profile must be set to a number between 0 "
                                 "and 1, or either to 'fixed' or 'infer'")

        r_2_1_guess = _core_guess(squared_R, mu, squared_radius, 0.9)
        s_guess = np.log(2) / (squared_radius - r_2_1_guess)

        if self.profile == "fixed":
            def r2_to_mu(R_2, r_2_1):
                return [exp_clip(-np.log(2) * \
                                 (r_2 - r_2_1) / (squared_radius - r_2_1))
                        for r_2 in R_2]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                upper = _core_upper(squared_radius)
                [r_2_1_opt], _ = curve_fit(
                    r2_to_mu, squared_R, mu,
                    p0=(min(max(r_2_1_guess, 0), upper),), maxfev=2000,
                    bounds=((0,), (upper,)))
                denominator = squared_radius - r_2_1_opt
                self.slope_ = -np.log(2) / denominator
                self.intercept_ = r_2_1_opt * np.log(2) / denominator

        elif self.profile == "infer":
            def r2_to_mu(R_2, r_2_1, s):
                return [exp_clip(-(r_2 - r_2_1) / s) for r_2 in R_2]

            try:
                p_opt, _ = curve_fit(r2_to_mu, squared_R, mu,
                                    p0=(r_2_1_guess, s_guess),
                                    # bounds=((0, 0), (np.inf, np.inf)),
                                    maxfev=2000)
                r_2_1_opt, s_opt = p_opt
                self.slope_ = -1 / s_opt
                self.intercept_ = r_2_1_opt / s_opt
            except RuntimeError:
                # interpolation could not take place, fall back to fixed profile
                self._fit_as_fixed(squared_R, mu, squared_radius)

        elif isinstance(self.profile, (int, float)):
            alpha = self.profile
            def r2_to_mu(R_2, r_2_1):
                inner = [r_2 - r_2_1 for r_2 in squared_R if r_2 > r_2_1]
                if len(inner) > 0:
                    q = np.percentile(inner, 100 * alpha)
                    return [exp_clip(np.log(alpha) / q * (r_2 - r_2_1))
                            for r_2 in R_2]
                else:
                    # all points have within the sphere -> unit membership
                    return [1] * len(R_2)

            try:
                [r_2_1_opt], _ = curve_fit(r2_to_mu, squared_R, mu,
                                        p0=(r_2_1_guess,),
                                        bounds=((0,), (np.inf,)))
                inner = [r_2 - r_2_1_opt for r_2 in squared_R if r_2 > r_2_1_opt]
                q = np.percentile(inner, 100 * alpha)
                self.slope_ = np.log(alpha) / q
                self.intercept_ = -r_2_1_opt * np.log(alpha) / q
            except RuntimeError:
                # interpolation could not take place, fall back to fixed profile
                self._fit_as_fixed(squared_R, mu, squared_radius)
            
        else:
            raise ValueError("'self.profile' attribute should be equal to "
                             "'infer', 'fixed' or 'alpha' "
                             f"(provided value: {self.profile})")
        
        if self.slope_ > 0:
            logger.warning('Profile fitting returned a positive slope '
                            f'({self.slope_})')
        if self.intercept_ < 0:
            logger.warning('Profile fitting returned a negative intercept '
                            f'({self.intercept_})')
        
        return self
    
    def get_membership(self, R_2):
        check_is_fitted(self, ['slope_', 'intercept_'])
        return exp_clip(self.slope_ * R_2 + self.intercept_)


def _quantile_knots(squared_R, mu, squared_radius, profile):
    """Compute the knots of the quantile-based piecewise fuzzifiers.

    Returns the tuple `(r_2_1, q1, m, q3, max)`: `r_2_1` is the median squared
    distance of the points having maximal membership, and the remaining values
    are offsets from `r_2_1`. With profile `'infer'` they are the quartiles and
    the maximum of the excess squared distances beyond `r_2_1`; with profile
    `'fixed'` the median knot is anchored to `squared_radius`, so that the
    membership function equals 1/2 there, `q1` is the median of the excess
    distances lower than `squared_radius`, and `q3` and `max` are the
    median and the maximum of those exceeding it."""

    if profile not in ('fixed', 'infer'):
        raise ValueError("'profile' parameter should either be equal to "
                         f"'fixed' or 'infer' (provided: {profile})")

    assert len(squared_R) == len(mu)

    r_2_1 = np.median([r_2 for r_2, m in zip(squared_R, mu)
                       if m >= max(mu)*0.99])

    if profile == 'infer':
        external_dist = [r_2 - r_2_1 for r_2 in squared_R if r_2 > r_2_1]
        if not external_dist:
            return r_2_1, 0, 0, 0, 0
        return (r_2_1, np.percentile(external_dist, 25),
                np.median(external_dist), np.percentile(external_dist, 75),
                np.max(external_dist))

    if r_2_1 >= squared_radius:
        logger.warning('Core of the fuzzy set extends beyond the sphere '
                       f'(squared radius {squared_radius}): core shrunk to '
                       'the center')
        r_2_1 = 0

    m = squared_radius - r_2_1
    inner = [r_2 - r_2_1 for r_2 in squared_R if r_2_1 < r_2 < squared_radius]
    outer = [r_2 - r_2_1 for r_2 in squared_R if r_2 > squared_radius]
    q1 = np.median(inner) if inner else m / 2
    q3, max_ = (np.median(outer), np.max(outer)) if outer else (m, m)
    return r_2_1, q1, m, q3, max_


class QuantileConstantPiecewiseFuzzifier(Fuzzifier):
    """Quantile-based constant piecewise fuzzifier.

    Fuzzifier corresponding to a fuzzy set with a piecewise constant membership
    function, whose steps are defined according to the quartiles of the squared
    distances between images of points and center of the learnt sphere."""

    default_profile = 'infer'
    def __init__(self, profile=default_profile):
        r"""Create an instance of :class:`QuantileConstantPiecewiseFuzzifier`.
        
        :param profile: method to be used in order to build the fuzzifier
          profile: `'infer'` places the steps at the quartiles of the squared
          distances lying outside the core, while `'fixed'` anchors the
          median step to the radius of the sphere, so that the membership
          is 1/2 there.
        :type profile: str"""

        super().__init__(profile)

    def fit(self, squared_R, mu, squared_radius):
        """Fit the fuzzifier on training data.

        :param squared_R: iterable of squared distance of the images of vectors
          in data space w.r.t. the center of the fuzzy set in feature space.
        :type squared_R: iterable of `float`
        :param mu: membership degrees of the vectors having originated
          `squared_R`.
        :type mu: vector of floats having the same length of `squared_R`
        :param squared_radius: radius of the fuzzy set in feature space.
        :type squared_radius: float
        
        :raises: ValueError if self.profile is not set either to `'fixed'`
          or to `'infer'`.

        The piecewise membership function is built so that its steps are chosen
        according to the quartiles of square distances between images of the
        points in `X` center of the learnt sphere.
        """

        self.name = 'QuantileConstPiecewise'
        self.latex_name = '$\\hat\\mu_{\\text{q\\_const}}$'

        self.r_2_1_, self.q1_, self.m_, self.q3_, _ = \
            _quantile_knots(squared_R, mu, squared_radius, self.profile)

        return self
    
    def get_membership(self, R_2):
        check_is_fitted(self, ['r_2_1_', 'm_', 'q1_', 'q3_'])
        return np.array([1 if r_2 <= self.r_2_1_ \
                         else 0.75 if r_2 <= self.r_2_1_ + self.q1_ \
                         else 0.5 if r_2 <= self.r_2_1_ + self.m_ \
                         else 0.25 if r_2 <= self.r_2_1_ + self.q3_ \
                         else 0 for r_2 in R_2])


class QuantileLinearPiecewiseFuzzifier(Fuzzifier):
    """Quantile-based linear piecewise fuzzifier.

    Fuzzifier corresponding to a fuzzy set with a piecewise linear membership
    function, whose steps are defined according to the quartiles of the squared
    distances between images of points and center of the learnt sphere."""

    default_profile = 'infer'
    def __init__(self, profile=default_profile):
        r"""Create an instance of :class:`QuantileLinearPiecewiseFuzzifier`.
        
        :param profile: method to be used in order to build the fuzzifier
          profile: `'infer'` places the steps at the quartiles of the squared
          distances lying outside the core, while `'fixed'` anchors the
          median step to the radius of the sphere, so that the membership
          is 1/2 there.
        :type profile: str"""

        super().__init__(profile)

    def fit(self, squared_R, mu, squared_radius):
        """Fit the fuzzifier on training data.

        :param squared_R: iterable of squared distance of the images of vectors
          in data space w.r.t. the center of the fuzzy set in feature space.
        :type squared_R: iterable of `float`
        :param mu: membership degrees of the vectors having originated
          `squared_R`.
        :type mu: vector of floats having the same length of `squared_R`
        :param squared_radius: radius of the fuzzy set in feature space.
        :type squared_radius: float

        :raises: ValueError if self.profile is not set either to `'fixed'`
          or to `'infer'`.

        The piecewise membership function is built so that its steps are chosen
        according to the quartiles of square distances between images of the
        points in `X` center of the learnt sphere.
        """

        self.name = 'QuantileLinPiecewise'
        self.latex_name = '$\\hat\\mu_{\\text{q\\_lin}}$'

        self.r_2_1_, self.q1_, self.m_, self.q3_, self.max_ = \
            _quantile_knots(squared_R, mu, squared_radius, self.profile)

        return self
        
    def get_membership(self, R_2):
        check_is_fitted(self, ['r_2_1_', 'm_', 'q1_', 'q3_', 'max_'])
        R_2 = np.asarray(R_2, dtype=float)
        knots = self.r_2_1_ + np.array([0, self.q1_, self.m_,
                                        self.q3_, self.max_])
        values = [1, .75, .5, .25, 0]

        # linear interpolation between consecutive knots; zero-width segments
        # are skipped, so that each knot gets the larger of the two values
        result = np.where(R_2 <= knots[0], 1., 0.)
        for k in range(4):
            if knots[k + 1] > knots[k]:
                inside = (R_2 > knots[k]) & (R_2 <= knots[k + 1])
                result[inside] = values[k] + (values[k + 1] - values[k]) * \
                    (R_2[inside] - knots[k]) / (knots[k + 1] - knots[k])
        return result


class SigmoidFuzzifier(Fuzzifier):
    """Sigmoid fuzzifier.

    Fuzzifier corresponding to a fuzzy set whose membership in feature space
    smoothly decreases from 1 to 0 following a logistic function."""

    default_profile = "fixed"
    def __init__(self, profile=default_profile):
        r"""Create an instance of :class:`SigmoidFuzzifier`.

        :param profile: method to be used in order to build the fuzzifier
          profile: `'fixed'` places the inflection point (where membership
          equals 0.5) at the squared radius of the learnt sphere and fits the
          slope, while `'infer'` fits both parameters on the provided examples.
        :type profile: str"""
        super().__init__(profile)

    def fit(self, squared_R, mu, squared_radius):
        r"""Fit the fuzzifier on training data.

        :param squared_R: iterable of squared distance of the images of vectors
          in data space w.r.t. the center of the fuzzy set in feature space.
        :type squared_R: iterable of `float`
        :param mu: membership degrees of the vectors having originated
          `squared_R`.
        :type mu: vector of floats having the same length of `squared_R`
        :param squared_radius: radius of the fuzzy set in feature space.
        :type squared_radius: float

        :raises: ValueError if self.profile is not set either to `'fixed'` or
          to `'infer'`.

        The membership profile has the form

        .. math::
          p(r) = \frac{1}{1 + \mathrm e^{(r - r_{0.5}) / s}} \enspace,

        where $s > 0$ controls the steepness of the transition. When `profile`
        is `'fixed'`, $r_{0.5}$ is set to `squared_radius`.
        """

        self.name = 'Sigmoid'
        self.latex_name = '$\\hat\\mu_{\\text{sigmoid}}$'

        assert len(squared_R) == len(mu)
        squared_R = np.asarray(squared_R, dtype=float)
        mu = np.asarray(mu, dtype=float)

        s_guess = max(np.ptp(squared_R) / 10, 1e-6)

        if self.profile == 'fixed':
            def r2_to_mu(R_2, s):
                return expit(-(R_2 - squared_radius) / s)

            try:
                [s_opt], _ = curve_fit(r2_to_mu, squared_R, mu,
                                       p0=(s_guess,),
                                       bounds=((1e-12,), (np.inf,)))
            except RuntimeError:
                logger.warning('Sigmoid profile fit failed, using the '
                               'initial guess for the steepness')
                s_opt = s_guess
            self.inflection_ = float(squared_radius)
            self.steepness_ = float(s_opt)

        elif self.profile == 'infer':
            def r2_to_mu(R_2, r_05, s):
                return expit(-(R_2 - r_05) / s)

            try:
                (r_05_opt, s_opt), _ = curve_fit(
                    r2_to_mu, squared_R, mu,
                    p0=(squared_radius, s_guess),
                    bounds=((-np.inf, 1e-12), (np.inf, np.inf)))
            except RuntimeError:
                logger.warning('Sigmoid profile fit failed, using the '
                               'initial guess for both parameters')
                r_05_opt, s_opt = squared_radius, s_guess
            self.inflection_ = float(r_05_opt)
            self.steepness_ = float(s_opt)

        else:
            raise ValueError("'profile' parameter should be equal to "
                             f"'fixed' or 'infer' (provided: {self.profile})")

        return self

    def get_membership(self, R_2):
        check_is_fitted(self, ['inflection_', 'steepness_'])
        R_2 = np.asarray(R_2, dtype=float)
        return expit(-(R_2 - self.inflection_) / self.steepness_)


class ZShapedFuzzifier(Fuzzifier):
    """Z-shaped fuzzifier.

    Fuzzifier corresponding to a fuzzy set whose membership in feature space
    decreases from 1 to 0 following Zadeh's Z-shaped (quadratic spline)
    function, i.e., a smooth version of :class:`LinearFuzzifier`."""

    default_profile = "fixed"
    def __init__(self, profile=default_profile):
        r"""Create an instance of :class:`ZShapedFuzzifier`.

        :param profile: method to be used in order to build the fuzzifier
          profile: `'fixed'` requires membership to be 0.5 at the squared
          radius of the learnt sphere and fits the beginning of the decreasing
          part, while `'infer'` fits both extremes of the decreasing part on
          the provided examples.
        :type profile: str"""
        super().__init__(profile)

    @staticmethod
    def _z(R_2, r_1, r_0):
        R_2 = np.asarray(R_2, dtype=float)
        width = max(r_0 - r_1, 1e-12)
        t = np.clip((R_2 - r_1) / width, 0, 1)
        return np.where(t <= 0.5, 1 - 2 * t ** 2, 2 * (1 - t) ** 2)

    def fit(self, squared_R, mu, squared_radius):
        r"""Fit the fuzzifier on training data.

        :param squared_R: iterable of squared distance of the images of vectors
          in data space w.r.t. the center of the fuzzy set in feature space.
        :type squared_R: iterable of `float`
        :param mu: membership degrees of the vectors having originated
          `squared_R`.
        :type mu: vector of floats having the same length of `squared_R`
        :param squared_radius: radius of the fuzzy set in feature space.
        :type squared_radius: float

        :raises: ValueError if self.profile is not set either to `'fixed'` or
          to `'infer'`.

        The membership profile equals 1 for $r \leq r_1$, 0 for $r > r_0$,
        and in between it is the quadratic spline

        .. math::
          p(r) = \begin{cases}
            1 - 2 t^2 & \text{if $t \leq 1/2$,} \\
            2 (1 - t)^2 & \text{otherwise,}
          \end{cases} \qquad t = \frac{r - r_1}{r_0 - r_1} \enspace.

        When `profile` is `'fixed'`, $r_0 = 2 r_{0.5} - r_1$, where $r_{0.5}$
        is `squared_radius`, so that the profile equals 0.5 at the squared
        radius of the sphere.
        """

        self.name = 'ZShaped'
        self.latex_name = '$\\hat\\mu_{\\text{Z}}$'

        assert len(squared_R) == len(mu)
        squared_R = np.asarray(squared_R, dtype=float)
        mu = np.asarray(mu, dtype=float)

        r_1_guess = np.median(squared_R[mu >= max(mu) * 0.99])
        r_0_guess = np.median(squared_R[mu <= min(mu) * 1.01])
        if r_0_guess <= r_1_guess:
            r_0_guess = r_1_guess + max(np.ptp(squared_R), 1e-6)

        if self.profile == 'fixed':
            upper = max(squared_radius * (1 - 1e-9), 1e-12)
            def r2_to_mu(R_2, r_1):
                return self._z(R_2, r_1, 2 * squared_radius - r_1)

            try:
                [r_1_opt], _ = curve_fit(
                    r2_to_mu, squared_R, mu,
                    p0=(np.clip(r_1_guess, 0, upper * 0.99),),
                    bounds=((0,), (upper,)))
            except RuntimeError:
                logger.warning('Z-shaped profile fit failed, using the '
                               'initial guess')
                r_1_opt = np.clip(r_1_guess, 0, upper)
            self.r_1_ = float(r_1_opt)
            self.r_0_ = float(2 * squared_radius - r_1_opt)

        elif self.profile == 'infer':
            def r2_to_mu(R_2, r_1, width):
                return self._z(R_2, r_1, r_1 + width)

            try:
                (r_1_opt, w_opt), _ = curve_fit(
                    r2_to_mu, squared_R, mu,
                    p0=(r_1_guess, r_0_guess - r_1_guess),
                    bounds=((-np.inf, 1e-12), (np.inf, np.inf)))
            except RuntimeError:
                logger.warning('Z-shaped profile fit failed, using the '
                               'initial guess')
                r_1_opt, w_opt = r_1_guess, r_0_guess - r_1_guess
            self.r_1_ = float(r_1_opt)
            self.r_0_ = float(r_1_opt + w_opt)

        else:
            raise ValueError("'profile' parameter should be equal to "
                             f"'fixed' or 'infer' (provided: {self.profile})")

        return self

    def get_membership(self, R_2):
        check_is_fitted(self, ['r_1_', 'r_0_'])
        return self._z(R_2, self.r_1_, self.r_0_)


class IsotonicFuzzifier(Fuzzifier):
    """Isotonic fuzzifier.

    Fuzzifier corresponding to the nonincreasing function of the squared
    distance in feature space which best fits the provided membership degrees
    in the least-squares sense, without any assumption on its shape."""

    default_profile = "infer"
    def __init__(self, profile=default_profile):
        r"""Create an instance of :class:`IsotonicFuzzifier`.

        :param profile: method to be used in order to build the fuzzifier
          profile; only `'infer'` is supported.
        :type profile: str"""
        super().__init__(profile)

    def fit(self, squared_R, mu, squared_radius):
        r"""Fit the fuzzifier on training data.

        :param squared_R: iterable of squared distance of the images of vectors
          in data space w.r.t. the center of the fuzzy set in feature space.
        :type squared_R: iterable of `float`
        :param mu: membership degrees of the vectors having originated
          `squared_R`.
        :type mu: vector of floats having the same length of `squared_R`
        :param squared_radius: radius of the fuzzy set in feature space (not
          used, kept for interface compatibility).
        :type squared_radius: float

        :raises: ValueError if self.profile is not set to `'infer'`.

        The profile is obtained through isotonic regression (pool adjacent
        violators algorithm) of `mu` against `squared_R`, constrained to be
        nonincreasing and to take values in $[0, 1]$; it is linearly
        interpolated between the fitted points, and constant before the first
        and after the last of them.
        """

        if self.profile != 'infer':
            raise ValueError("'profile' parameter should be equal to "
                             f"'infer' (provided: {self.profile})")

        self.name = 'Isotonic'
        self.latex_name = '$\\hat\\mu_{\\text{iso}}$'

        assert len(squared_R) == len(mu)

        iso = IsotonicRegression(increasing=False, y_min=0, y_max=1,
                                 out_of_bounds='clip')
        iso.fit(np.asarray(squared_R, dtype=float),
                np.asarray(mu, dtype=float))
        # stored as tuples, so that fitted fuzzifiers can be compared via ==
        self.knots_ = tuple(float(v) for v in iso.X_thresholds_)
        self.values_ = tuple(float(v) for v in iso.y_thresholds_)

        return self

    def get_membership(self, R_2):
        check_is_fitted(self, ['knots_', 'values_'])
        return np.interp(np.asarray(R_2, dtype=float),
                         self.knots_, self.values_)
