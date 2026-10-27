import json
import numpy as np
import os
import pickle
import unittest
from unittest import mock

from sklearn.datasets import load_iris

from mulearn import FuzzyInductor
import mulearn.fuzzifier as fuzz


class BaseTest:
    def setUp(self):
        d = load_iris()
        X = d['data']
        self.X = X[:, 2]
        self.y = np.array([1] * 50 + [0] * 100)
    
    def _test_compute(self, fuzzifier, squared_R, mu, squared_radius, target):
        fuzzifier.fit(squared_R, mu, squared_radius)
        if isinstance(fuzzifier, fuzz.CrispFuzzifier):
            self.assertTrue(set(fuzzifier.get_membership(squared_R)) <= {0, 1})
        else:
            self.assertIsInstance(fuzzifier.slope_, (int, float))
            self.assertIsInstance(fuzzifier.intercept_, (int, float))
            self.assertTrue(fuzzifier.slope_ < 0)
            self.assertTrue(fuzzifier.intercept_ > 0)

        result = fuzzifier.get_membership(squared_R)
        for t, r in zip(target, result):
            self.assertAlmostEqual(t, r)

    def _test_profile(self, fuzzifier, attributes, values):
         fuzzifier.fit(self.X, self.y, 3.5)
         for a, v in zip(attributes, values):
             self.assertAlmostEqual(fuzzifier.__getattribute__(a), v, places=5)
    
         _, _ ,_ = fuzzifier.get_profile(self.X)
    
    def _test_serialize(self, fuzzifier):
        fuzzifier.fit(self.X, self.y, 3.5)
        s = pickle.dumps(fuzzifier)  
        fuzzifier_clone = pickle.loads(s)
        self.assertEqual(fuzzifier, fuzzifier_clone)

    def _test_persist(self, fuzzifier):
        fuzzifier.fit(self.X, self.y, 3.5)

        with open('object.pickle', 'wb') as f:
            pickle.dump(fuzzifier, f)

        with open('object.pickle', 'rb') as f:
            fuzzifier_clone = pickle.load(f)

        os.remove('object.pickle')

        self.assertEqual(fuzzifier, fuzzifier_clone)
    
    def _test_json(self, fuzzifier, target):
        s = json.dumps(fuzzifier)
        repr = json.loads(s)
        self.assertEqual(repr, target)


class TestCrispFuzzifier(BaseTest, unittest.TestCase):

    def test_compute(self):
        squared_R = np.array([1, 3, 6, 12, 34, 416])
        mu = np.array([1, .9, .5, .3, .2, .1])

        self._test_compute(fuzz.CrispFuzzifier(profile='fixed'),
                           squared_R, mu, 2, np.array([1, 0, 0, 0, 0, 0]))
        
        self._test_compute(fuzz.CrispFuzzifier(profile='infer'),
                           squared_R, mu, 2, np.array([1, 1, 0, 0, 0, 0]))

    def test_infer_is_optimal(self):
        rng = np.random.default_rng(0)
        squared_R = np.round(rng.uniform(0, 10, 60), 1)
        mu = np.clip(1 - squared_R / 6 + rng.normal(0, .2, 60), 0, 1)
        f = fuzz.CrispFuzzifier('infer').fit(squared_R, mu, 5)
        sse = lambda t: np.sum(((squared_R <= t) - mu) ** 2)
        best = min(sse(t) for t in np.r_[-1, squared_R])
        self.assertAlmostEqual(sse(f.threshold_), best)

    def test_infer_extremes(self):
        squared_R = np.array([1., 2, 3])
        f = fuzz.CrispFuzzifier('infer')
        self.assertEqual(f.fit(squared_R, [1, 1, 1], 2).threshold_, 3)
        self.assertEqual(list(f.fit(squared_R, [0, 0, 0], 2)
                              .get_membership(squared_R)), [0, 0, 0])

    def test_profile(self):
        for profile, threshold in zip(['fixed', 'infer'], (3.5, 2.45)):
            self._test_profile(fuzz.CrispFuzzifier(profile=profile),
                               ['threshold_'], [threshold])
    
    def test_serialize(self):
        for profile in ['fixed', 'infer']:
            self._test_serialize(fuzz.CrispFuzzifier(profile=profile))

    def test_persist(self):
        for profile in ['fixed', 'infer']:
            self._test_persist(fuzz.CrispFuzzifier(profile=profile))
    
    def test_json(self):
        for profile in ['fixed', 'infer']:
            self._test_json(fuzz.CrispFuzzifier(profile=profile),
                        {'class': 'CrispFuzzifier', 'profile': profile})


class TestLinearFuzzifier(BaseTest, unittest.TestCase):

    def test_compute(self):
        squared_R = np.array([1, 3, 6, 12, 34, 416])
        mu = np.array([1, .9, .5, .3, .2, .1])
        self._test_compute(fuzz.LinearFuzzifier(profile='fixed'),
                           squared_R, mu, 2, np.array([1, 0, 0, 0, 0, 0]))
        target = [0.99763019, 0.99289056, 0.98578113,
                  0.97156225, 0.91942638, 0.01415807]
        self._test_compute(fuzz.LinearFuzzifier(profile='triangular'),
                           squared_R, mu, 2, target)
        target = np.array([0.60031447, 0.59773126, 0.59385645,
                           0.58610684, 0.55769158, 0.06429942])
        self._test_compute(fuzz.LinearFuzzifier(profile='infer'),
                           squared_R, mu, 2, target)

    def test_profile(self):
        slopes = (-0.7926829266641029, -0.9984776805603038)
        intercepts = (3.27439024332436, 2.9373025049777697)
        
        for profile, slope, intercept in zip(['fixed', 'infer'],
                                slopes, intercepts):

            self._test_profile(fuzz.LinearFuzzifier(profile=profile),
                               ['slope_', 'intercept_'],
                               [slope, intercept])
    
    def test_serialize(self):
        for profile in ['fixed', 'infer']:
            self._test_serialize(fuzz.LinearFuzzifier(profile=profile))

    def test_persist(self):
        for profile in ['fixed', 'infer']:
            self._test_persist(fuzz.LinearFuzzifier(profile=profile))
    
    def test_json(self):
        for profile in ['fixed', 'infer']:
            self._test_json(fuzz.LinearFuzzifier(profile=profile),
                        {'class': 'LinearFuzzifier', 'profile': profile})

        
class TestExponentialFuzzifier(BaseTest, unittest.TestCase):

    def test_compute(self):
        squared_R = np.array([1, 3, 6, 12, 34, 37])
        mu = np.array([1, .9, .8, .7, .2, .1])
        target = np.array([0.84089642, 0.59460356, 0.35355339, 0.125     , 0.00276214,
       0.00164238])
        self._test_compute(fuzz.ExponentialFuzzifier(profile='fixed'),
                           squared_R, mu, 4, target)
        target = np.array([1.        , 0.93790848, 0.80971391,
                           0.60349527, 0.20539768, 0.17732365])
        self._test_compute(fuzz.ExponentialFuzzifier(profile='infer'),
                           squared_R, mu, 4, target)

        target = [
            np.array([1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                      6.99944998e-01, 4.16624055e-05, 1.10551439e-05]),
            np.array([1.00000000e+00, 1.00000000e+00, 7.54962443e-01,
                      8.24960386e-02, 2.46014950e-05, 8.13233710e-06]),
            np.array([1.00000000e+00, 8.81800354e-01, 4.08231664e-01,
                      8.74942523e-02, 3.08493908e-04, 1.42818021e-04]),
            np.array([1., 0.89834481, 0.62190419,
                      0.29804671, 0.02009127, 0.01390874]),
            np.array([1., 0.90433429, 0.74223711,
                      0.5, 0.11745466, 0.09640153]),
            np.array([1., 0.97403053, 0.89765262,
                      0.76239452, 0.41889818, 0.38605058]),
            np.array([1., 0.96809506, 0.92213607,
                      0.83666002, 0.58566202, 0.55785852]),
            np.array([0.99345844, 0.98050343, 0.96138697,
                      0.92426491, 0.8       , 0.78440274]),
            np.array([0.9970365 , 0.99113582, 0.9823502 ,
                      0.96501192, 0.9040156 , 0.89600224])
        ]

        for alpha, t in zip(np.linspace(0.1, 0.9, 9), target):
            self._test_compute(fuzz.ExponentialFuzzifier(profile=alpha),
                               squared_R, mu, 4, t)

    def test_profile(self):

        slopes = (-2.3887064381585272, -14.305178809577578,
                  -1.142022483767279, -0.32724662996583)
        intercepts = (7.6673253529949, 27.17983975471548,
                      2.1513025936983423, 0.5563191029720798)
        
        for profile, slope, intercept in zip(['fixed', 'infer', 0.1, 0.4],
                                slopes, intercepts):

            self._test_profile(fuzz.ExponentialFuzzifier(profile=profile),
                               ['slope_', 'intercept_'],
                               [slope, intercept])

    def test_serialize(self):
        for profile in ['fixed', 'infer', 0.01, 0.4]:
            self._test_serialize(fuzz.ExponentialFuzzifier(profile=profile))

    def test_persist(self):
        for profile in ['fixed', 'infer', 0.01, 0.4]:
            self._test_persist(fuzz.ExponentialFuzzifier(profile=profile))
    
    def test_json(self):
        for profile in ['fixed', 'infer', 0.01, 0.4]:
            self._test_json(fuzz.ExponentialFuzzifier(profile=profile),
                        {'class': 'ExponentialFuzzifier', 'profile': profile})


if __name__ == '__main__':
    unittest.main()


class SmoothFuzzifierTest(BaseTest):
    """Checks shared by the fuzzifiers which do not expose slope/intercept."""

    def _check_valid(self, fuzzifier, squared_R):
        grid = np.linspace(0, max(squared_R) * 1.5, 500)
        values = fuzzifier.get_membership(grid)
        self.assertTrue(np.all(values >= 0) and np.all(values <= 1))
        self.assertTrue(np.all(np.diff(values) <= 1e-12))


class TestSigmoidFuzzifier(SmoothFuzzifierTest, unittest.TestCase):

    def test_fixed(self):
        squared_R = np.array([1, 3, 6, 12, 34, 416])
        mu = np.array([1, .9, .5, .3, .2, .1])
        f = fuzz.SigmoidFuzzifier(profile='fixed').fit(squared_R, mu, 6)
        self.assertAlmostEqual(f.get_membership([6])[0], 0.5)
        self.assertTrue(f.steepness_ > 0)
        self._check_valid(f, squared_R)

    def test_infer_recovers_parameters(self):
        squared_R = np.linspace(0, 10, 50)
        mu = 1 / (1 + np.exp((squared_R - 4) / 0.7))
        f = fuzz.SigmoidFuzzifier(profile='infer').fit(squared_R, mu, 6)
        self.assertAlmostEqual(f.inflection_, 4, places=4)
        self.assertAlmostEqual(f.steepness_, 0.7, places=4)
        self._check_valid(f, squared_R)

    def test_wrong_profile(self):
        with self.assertRaises(ValueError):
            fuzz.SigmoidFuzzifier(profile='foo').fit([1, 2], [1, 0], 1.5)

    def test_serialize(self):
        for profile in ['fixed', 'infer']:
            self._test_serialize(fuzz.SigmoidFuzzifier(profile=profile))

    def test_persist(self):
        for profile in ['fixed', 'infer']:
            self._test_persist(fuzz.SigmoidFuzzifier(profile=profile))

    def test_json(self):
        for profile in ['fixed', 'infer']:
            self._test_json(fuzz.SigmoidFuzzifier(profile=profile),
                        {'class': 'SigmoidFuzzifier', 'profile': profile})


class TestZShapedFuzzifier(SmoothFuzzifierTest, unittest.TestCase):

    def test_shape(self):
        z = fuzz.ZShapedFuzzifier._z
        r = np.array([0, 1, 1.5, 2, 2.5, 3, 4])
        np.testing.assert_allclose(z(r, 1, 3),
                                   [1, 1, 0.875, 0.5, 0.125, 0, 0])

    def test_fixed(self):
        squared_R = np.array([1, 3, 6, 12, 34, 416])
        mu = np.array([1, .9, .5, .3, .2, .1])
        f = fuzz.ZShapedFuzzifier(profile='fixed').fit(squared_R, mu, 6)
        self.assertAlmostEqual(f.get_membership([6])[0], 0.5)
        self.assertTrue(0 <= f.r_1_ < 6 < f.r_0_)
        self._check_valid(f, squared_R)

    def test_infer_recovers_parameters(self):
        squared_R = np.linspace(0, 10, 60)
        mu = fuzz.ZShapedFuzzifier._z(squared_R, 2, 7)
        f = fuzz.ZShapedFuzzifier(profile='infer').fit(squared_R, mu, 5)
        self.assertAlmostEqual(f.r_1_, 2, places=3)
        self.assertAlmostEqual(f.r_0_, 7, places=3)
        self._check_valid(f, squared_R)

    def test_wrong_profile(self):
        with self.assertRaises(ValueError):
            fuzz.ZShapedFuzzifier(profile='foo').fit([1, 2], [1, 0], 1.5)

    def test_serialize(self):
        for profile in ['fixed', 'infer']:
            self._test_serialize(fuzz.ZShapedFuzzifier(profile=profile))

    def test_persist(self):
        for profile in ['fixed', 'infer']:
            self._test_persist(fuzz.ZShapedFuzzifier(profile=profile))

    def test_json(self):
        for profile in ['fixed', 'infer']:
            self._test_json(fuzz.ZShapedFuzzifier(profile=profile),
                        {'class': 'ZShapedFuzzifier', 'profile': profile})


class TestIsotonicFuzzifier(SmoothFuzzifierTest, unittest.TestCase):

    def test_pool_adjacent_violators(self):
        squared_R = np.array([1, 2, 3, 4])
        mu = np.array([1, .2, .6, 0])
        f = fuzz.IsotonicFuzzifier().fit(squared_R, mu, 2.5)
        np.testing.assert_allclose(f.get_membership(squared_R),
                                   [1, .4, .4, 0])

    def test_monotone_data_fitted_exactly(self):
        squared_R = np.array([1, 3, 6, 12, 34, 416])
        mu = np.array([1, .9, .5, .3, .2, .1])
        f = fuzz.IsotonicFuzzifier().fit(squared_R, mu, 6)
        np.testing.assert_allclose(f.get_membership(squared_R), mu)
        self.assertEqual(f.get_membership([0])[0], 1)
        self.assertAlmostEqual(f.get_membership([1000])[0], .1)
        self._check_valid(f, squared_R)

    def test_not_worse_than_other_monotone_fuzzifiers(self):
        rng = np.random.default_rng(0)
        squared_R = np.sort(rng.uniform(0, 10, 80))
        mu = np.clip(1 - squared_R / 8 + rng.normal(0, .15, 80), 0, 1)
        iso = fuzz.IsotonicFuzzifier().fit(squared_R, mu, 4)
        sse = lambda f: np.sum((f.get_membership(squared_R) - mu) ** 2)
        for other in (fuzz.LinearFuzzifier('infer'),
                      fuzz.SigmoidFuzzifier('infer'),
                      fuzz.ZShapedFuzzifier('infer')):
            other.fit(squared_R, mu, 4)
            self.assertLessEqual(sse(iso), sse(other) + 1e-9)

    def test_wrong_profile(self):
        with self.assertRaises(ValueError):
            fuzz.IsotonicFuzzifier(profile='fixed').fit([1, 2], [1, 0], 1.5)

    def test_serialize(self):
        self._test_serialize(fuzz.IsotonicFuzzifier())

    def test_persist(self):
        self._test_persist(fuzz.IsotonicFuzzifier())

    def test_json(self):
        self._test_json(fuzz.IsotonicFuzzifier(),
                        {'class': 'IsotonicFuzzifier', 'profile': 'infer'})


class TestQuantileFuzzifiers(SmoothFuzzifierTest, unittest.TestCase):

    classes = (fuzz.QuantileConstantPiecewiseFuzzifier,
               fuzz.QuantileLinearPiecewiseFuzzifier)

    def setUp(self):
        super().setUp()
        self.squared_R = np.arange(1, 11)
        self.mu = np.array([1, 1, .8, .7, .6, .4, .3, .2, .1, 0])

    def _knots(self, f):
        return [f.r_2_1_, f.q1_, f.m_, f.q3_]

    def test_infer(self):
        for cls in self.classes:
            f = cls('infer').fit(self.squared_R, self.mu, 5)
            np.testing.assert_allclose(self._knots(f), [1.5, 2.5, 4.5, 6.5])
            self._check_valid(f, self.squared_R)

    def test_fixed(self):
        for cls in self.classes:
            f = cls('fixed').fit(self.squared_R, self.mu, 5)
            np.testing.assert_allclose(self._knots(f), [1.5, 1.5, 3.5, 6.5])
            self.assertAlmostEqual(f.get_membership([5])[0], .5)
            self._check_valid(f, self.squared_R)
        self.assertEqual(f.max_, 8.5)

    def test_linear_continuous(self):
        for profile in ('fixed', 'infer'):
            f = fuzz.QuantileLinearPiecewiseFuzzifier(profile)
            f.fit(self.squared_R, self.mu, 5)
            knots = f.r_2_1_ + np.array([0, f.q1_, f.m_, f.q3_, f.max_])
            np.testing.assert_allclose(f.get_membership(knots),
                                       [1, .75, .5, .25, 0])
            np.testing.assert_allclose(f.get_membership(knots - 1e-9),
                                       f.get_membership(knots + 1e-9),
                                       atol=1e-6)

    def test_fixed_core_beyond_radius(self):
        for cls in self.classes:
            with self.assertLogs('mulearn.fuzzifier', level='WARNING'):
                f = cls('fixed').fit(self.squared_R, self.mu, 1)
            self.assertEqual(f.r_2_1_, 0)
            self.assertAlmostEqual(f.get_membership([1])[0], .5)

    def test_wrong_profile(self):
        for cls in self.classes:
            with self.assertRaises(ValueError):
                cls(profile='wrong').fit(self.squared_R, self.mu, 5)

    def test_serialize(self):
        for cls in self.classes:
            self._test_serialize(cls('fixed'))

    def test_persist(self):
        for cls in self.classes:
            self._test_persist(cls('fixed'))

    def test_json(self):
        for cls in self.classes:
            self._test_json(cls('fixed'),
                            {'class': cls.__name__, 'profile': 'fixed'})


class TestBaseFuzzifier(unittest.TestCase):

    def test_abstract_methods(self):
        f = fuzz.Fuzzifier('fixed')
        with self.assertRaises(NotImplementedError):
            f.fit([1, 2], [1, 0], 1.5)
        with self.assertRaises(NotImplementedError):
            f.get_membership([1, 2])


class TestFitFailure(unittest.TestCase):

    def test_fallback_keeps_profile(self):
        squared_R = np.array([1, 3, 6, 12, 34, 416])
        mu = np.array([1, .9, .5, .3, .2, .1])
        real_curve_fit = fuzz.curve_fit

        for profile, cls in (('infer', fuzz.LinearFuzzifier),
                             ('triangular', fuzz.LinearFuzzifier),
                             ('infer', fuzz.ExponentialFuzzifier),
                             (.3, fuzz.ExponentialFuzzifier)):
            fixed = cls('fixed').fit(squared_R, mu, 6)
            calls = []

            def fail_first_call(*args, **kwargs):
                # only the fit with the requested profile fails
                calls.append(None)
                if len(calls) == 1:
                    raise RuntimeError
                return real_curve_fit(*args, **kwargs)

            f = cls(profile)
            with mock.patch.object(fuzz, 'curve_fit', fail_first_call), \
                 self.assertLogs('mulearn.fuzzifier', level='WARNING'):
                f.fit(squared_R, mu, 6)
            self.assertEqual(f.profile, profile)
            self.assertAlmostEqual(f.slope_, fixed.slope_)
            self.assertAlmostEqual(f.intercept_, fixed.intercept_)


class TestNewFuzzifiersInInductor(unittest.TestCase):

    def test_fit_predict(self):
        d = load_iris()
        X = d['data'][:, 2:]
        y = np.array([1] * 50 + [0] * 100)
        for f in (fuzz.SigmoidFuzzifier('fixed'),
                  fuzz.SigmoidFuzzifier('infer'),
                  fuzz.ZShapedFuzzifier('fixed'),
                  fuzz.ZShapedFuzzifier('infer'),
                  fuzz.IsotonicFuzzifier()):
            model = FuzzyInductor(c=1, fuzzifier=f).fit(X, y)
            mu = model.predict(X)
            self.assertTrue(np.all((mu >= 0) & (mu <= 1)))
            self.assertGreater(np.mean(mu[:50]), np.mean(mu[50:]))
