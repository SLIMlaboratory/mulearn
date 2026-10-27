import unittest

import numpy as np

from mulearn.kernel import GaussianKernel, LinearKernel
from mulearn.shadowed import ShadowedSetInductor, SHADOW


class TestShadowedSetInductor(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1.], [2], [3], [4], [10], [11], [12], [13]])
        self.mu = np.array([0., 0, 1, 1, 1, 1, 1, 0])
        rng = np.random.default_rng(1)
        self.X_random = rng.normal(size=(25, 2))
        self.mu_random = np.clip(rng.uniform(-.3, 1.3, 25), 0, 1)

    def _primal_and_dual(self, model, X, mu):
        weights = model.chis_ + model.betas_
        gram = model.gram_
        dual = weights @ np.diag(gram) - weights @ gram @ weights
        sq_dist = model.x_to_sq_dist(X)
        r = model.sq_radius_
        primal = (r + model.d * (model.theta_0_ - model.theta_1_) +
                  model.c * np.sum(mu * np.maximum(sq_dist - r, 0) +
                                   (1 - mu) * np.maximum(r - sq_dist, 0)))
        return primal, dual

    def test_strong_duality(self):
        cases = [(self.X, self.mu, LinearKernel(), 0.2),
                 (self.X_random, self.mu_random, GaussianKernel(1), 0.5)]
        for X, mu, k, c in cases:
            for d in (0.01, 0.1, 0.25):
                model = ShadowedSetInductor(c=c, d=d, k=k).fit(X, mu)
                primal, dual = self._primal_and_dual(model, X, mu)
                self.assertAlmostEqual(primal, dual, delta=1e-4)

    def test_constraints(self):
        for d in (0.01, 0.1, 0.25):
            model = ShadowedSetInductor(c=0.2, d=d,
                                        k=LinearKernel()).fit(self.X, self.mu)
            self.assertAlmostEqual(np.sum(model.chis_ + model.betas_), 1)
            self.assertLessEqual(np.sum(model.betas_), 2 * d + 1e-8)
            self.assertLessEqual(model.theta_1_, model.theta_0_)
            self.assertLessEqual(model.x_to_sq_dist(self.X).max(),
                                 model.theta_0_ + 1e-8)

    def test_membership(self):
        model = ShadowedSetInductor(c=1, d=0.25,
                                    k=LinearKernel()).fit(self.X, self.mu)
        grid = np.linspace(0, 15, 151)[:, None]
        mu = model.fuzzy_membership(grid)
        self.assertTrue(np.all((mu >= 0) & (mu <= 1)))
        # membership is a nonincreasing function of the squared distance
        order = np.argsort(model.x_to_sq_dist(grid))
        self.assertTrue(np.all(np.diff(mu[order]) <= 1e-12))
        self.assertTrue(set(model.predict(grid)) <= {0, SHADOW, 1})

    def test_infeasible(self):
        with self.assertRaises(ValueError):
            ShadowedSetInductor(c=0.01, d=0.1).fit(self.X, self.mu)

    def _three_way_primal(self, model, X, mu):
        sq_dist = model.x_to_sq_dist(X)
        return (model.theta_0_ + model.d * (model.theta_0_ - model.theta_1_)
                + model.c * np.sum(mu * np.maximum(sq_dist - model.theta_0_, 0)
                                   + (1 - mu) * np.maximum(model.theta_1_ -
                                                           sq_dist, 0)))

    def test_three_way_strong_duality(self):
        for d in (0, 0.05, 0.2, 1):
            model = ShadowedSetInductor(c=0.5, d=d, k=GaussianKernel(1),
                                        formulation='three-way')
            model.fit(self.X_random, self.mu_random)
            weights, gram = model.weights_, model.gram_
            dual = weights @ np.diag(gram) - weights @ gram @ weights
            primal = self._three_way_primal(model, self.X_random,
                                            self.mu_random)
            self.assertAlmostEqual(primal, dual, delta=1e-5)
            self.assertLessEqual(model.theta_1_, model.theta_0_)
            self.assertLessEqual(np.sum(model.q_), d + 1e-8)

    def test_three_way_reduces_to_mulearn(self):
        from mulearn import FuzzyInductor
        from mulearn.fuzzifier import CrispFuzzifier
        from mulearn.optimization import ClarabelSolver
        model = ShadowedSetInductor(c=0.5, d=10, k=GaussianKernel(1),
                                    formulation='three-way')
        model.fit(self.X_random, self.mu_random)
        inductor = FuzzyInductor(c=0.5, k=GaussianKernel(1),
                                 fuzzifier=CrispFuzzifier(),
                                 solver=ClarabelSolver())
        inductor.fit(self.X_random, self.mu_random)
        self.assertAlmostEqual(model.theta_0_, model.theta_1_)
        self.assertAlmostEqual(model.theta_0_,
                               inductor.fuzzifier.sq_radius_05, delta=1e-5)

    def test_three_way_separable(self):
        # no overlap between positive and negative points: no shadow
        model = ShadowedSetInductor(c=1, d=0.05, k=LinearKernel(),
                                    formulation='three-way')
        model.fit(self.X, self.mu)
        self.assertEqual(model.theta_0_, model.theta_1_)
        np.testing.assert_array_equal(model.predict(self.X), self.mu)

    def test_unknown_formulation(self):
        with self.assertRaises(ValueError):
            ShadowedSetInductor(formulation='other').fit(self.X, self.mu)


if __name__ == '__main__':
    unittest.main()
