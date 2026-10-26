
import numpy as np
import unittest

from mulearn import FuzzyInductor
from mulearn.kernel import GaussianKernel
from mulearn.optimization import ClarabelSolver, GurobiSolver, nearest_psd


class TestGurobiSolver(unittest.TestCase):

    def testCompute(self):

        s = GurobiSolver()

        xs = np.array([[0.2526861], [0.77908776], [0.5120937],
                       [0.52646533], [0.01438627]])

        mus = np.array([0.63883086, 0.56515446, 0.99892903,
                        0.99488161, 0.17768801])

        c = 1

        #obtained from the original mulearn module
        chis_opt = [0.26334825774012194, 0.5651531004941153,
                    -0.0010709737624955377, -0.00511839469274798,
                    0.17768801022078584]
        
        k = np.array([[1.       , 0.8706202, 0.9669135, 0.9632160, 0.9720059],
                      [0.8706202, 1.       , 0.9649848, 0.9685946, 0.7464816],
                      [0.9669135, 0.9649848, 1.       , 0.9998967, 0.8835067],
                      [0.9632160, 0.9685946, 0.9998967, 1.       , 0.8771191],
                      [0.9720059 ,0.7464816, 0.8835067, 0.8771191, 1.       ]])

        for chi, chi_opt in zip(s.solve(xs, mus, c, k), chis_opt):
            self.assertAlmostEqual(chi, chi_opt, places=5)


        self.assertEqual(GurobiSolver().__repr__(), 'GurobiSolver()')
        
        self.assertEqual(GurobiSolver(adjustment='auto').__repr__(),
                         'GurobiSolver(adjustment=auto)')

        self.assertEqual(GurobiSolver(time_limit=1000).__repr__(),
                         'GurobiSolver(time_limit=1000)')

        self.assertEqual(GurobiSolver(initial_values=(1,2,3,4,5)).__repr__(),
                         'GurobiSolver(initial_values=(1, 2, 3, 4, 5))')

        self.assertEqual(GurobiSolver(initial_values=(1,2,3,4,5),
                                      adjustment='auto').__repr__(),
            'GurobiSolver(adjustment=auto, initial_values=(1, 2, 3, 4, 5))')

        
        model = FuzzyInductor()
        xs = np.array([
            [0.0529931],[0.0083108],[0.5267421],[0.2486910],[0.0251151],
            [0.1652516],[0.6417665],[0.1196049],[0.9794178],[0.1612251],
            [0.4119931],[0.3356147],[0.4766746],[0.7740397],[0.6974139],
            [0.0767959],[0.4183671],[0.0911699],[0.8647524],[0.7957013],
            [0.6492061],[0.0777168],[0.0107658],[0.3432774],[0.5856382],
            [0.2890628],[0.0132868],[0.4368564],[0.6674424],[0.6966816],
            [0.0819116],[0.2468026],[0.3826838],[0.8054723],[0.4339475],
            [0.8033825],[0.8945941],[0.3637644],[0.8285576],[0.1784889],
            [0.1342601],[0.8580456],[0.2053696],[0.5880164],[0.1460810],
            [0.1117827],[0.5497901],[0.0229859],[0.2912594],[0.1610982]])

        mus = np.array([0.0085146, 0.0031310, 0.9830871, 0.2217082, 0.0046125,
                        0.0690616, 0.6191740, 0.0317020, 0.0041603, 0.0647358,
                        0.8313222, 0.5249052, 0.9871067, 0.1667561, 0.3947267,
                        0.0139551, 0.8530416, 0.0185618, 0.0418628, 0.1242336,
                        0.5880158, 0.0142166, 0.0033161, 0.5566333, 0.8395180,
                        0.3460139, 0.0035165, 0.9092821, 0.5123567, 0.3974534,
                        0.0154637, 0.2167269, 0.7201648, 0.1079920, 0.9011676,
                        0.1113196, 0.0243836, 0.6423038, 0.0761680, 0.0849619,
                        0.0411487, 0.0469942, 0.1261214, 0.8312888, 0.0504056,
                        0.0274667, 0.9425840, 0.0043948, 0.3537064, 0.0646031])

        chis_opt = np.array(
            [ 0.0085146,  0.0031310, -0.0169128,  0.2217082,  0.0046125,
              0.0690616, -0.3808259,  0.0317020,  0.0041603,  0.0647358,
             -0.1686777,  0.0723333, -0.0128932,  0.1667561,  0.3947267,
              0.0139551, -0.1469583,  0.0185618,  0.0418628,  0.1242336,
             -0.4119841,  0.0142166,  0.0033161, -0.4433666, -0.1604819,
              0.3460139,  0.0035165, -0.0907178,  0.4388794,  0.3974534,
              0.0154637,  0.2167269, -0.2798351,  0.1079920, -0.0988323,
              0.1113196,  0.0243836, -0.3576962,  0.0761680,  0.0849619,
              0.0411487,  0.0469942,  0.1261214, -0.1687111,  0.0504056,
              0.0274667, -0.0574159,  0.0043948,  0.3537064,  0.0646031])

        model.fit(xs, mus)

        for chi, chi_opt in zip(model.chis_, chis_opt):
            self.assertAlmostEqual(chi, chi_opt, places=5)


if __name__ == '__main__':
    unittest.main()
   

class TestClarabelSolver(unittest.TestCase):

    def _problem(self, seed, n=40, sigma=1.):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2))
        mu = np.clip(rng.uniform(-.3, 1.3, n), 0, 1)
        K = np.array([GaussianKernel(sigma).compute(x, X) for x in X])
        return X, mu, K

    def test_same_optimum_as_gurobi(self):
        objective = lambda chi, K: chi @ K @ chi - np.diag(K) @ chi
        for seed, sigma in enumerate([.1, 1, 10]):
            X, mu, K = self._problem(seed, sigma=sigma)
            c = 2 / mu.sum()
            chi_c = np.array(ClarabelSolver().solve(X, mu, c, K))
            chi_g = np.array(GurobiSolver().solve(X, mu, c, K))
            self.assertAlmostEqual(objective(chi_c, K), objective(chi_g, K),
                                   places=6)
            self.assertAlmostEqual(chi_c.sum(), 1)

    def test_infeasible(self):
        X, mu, K = self._problem(0)
        with self.assertRaises(RuntimeError):
            ClarabelSolver().solve(X, mu, .5 / mu.sum(), K)

    def test_inductor(self):
        X, mu, _ = self._problem(1)
        fi = FuzzyInductor(c=1, solver=ClarabelSolver()).fit(X, mu)
        fg = FuzzyInductor(c=1, solver=GurobiSolver()).fit(X, mu)
        np.testing.assert_allclose(fi.predict(X), fg.predict(X), atol=1e-4)

    def test_nearest_psd(self):
        _, _, K = self._problem(2, sigma=10)
        K_psd, is_psd = nearest_psd(K)
        self.assertTrue(is_psd)
        self.assertGreaterEqual(np.linalg.eigvalsh(K_psd)[0], 0)
        np.testing.assert_allclose(K_psd, K, atol=1e-10)

    def test_not_psd(self):
        K = np.array([[1., 2], [2, 1]])
        K_out, is_psd = nearest_psd(K)
        self.assertFalse(is_psd)
        np.testing.assert_array_equal(K_out, K)
        with self.assertRaises(ValueError):
            ClarabelSolver().solve(np.zeros((2, 1)), np.array([1., 1]), 1, K)
