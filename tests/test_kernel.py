import json
import numpy as np
import os
import pickle
import unittest

import mulearn.kernel as kernel

class BaseTest:
    def _test_serialize(self, kernel):
        s = pickle.dumps(kernel)  
        kernel_clone = pickle.loads(s)
        self.assertEqual(kernel, kernel_clone)
    
    def _test_persist(self, kernel):
        with open('object.pickle', 'wb') as f:
            pickle.dump(kernel, f)

        with open('object.pickle', 'rb') as f:
            kernel_clone = pickle.load(f)

        os.remove('object.pickle')

        self.assertEqual(kernel, kernel_clone)
    
    def _test_json(self, kernel, target):
        s = json.dumps(kernel)
        repr = json.loads(s)
        self.assertEqual(repr, target)


class Test_LinearKernel(BaseTest, unittest.TestCase):
    def test_compute(self):
        k =kernel.LinearKernel()
        self.assertEqual(k.compute(np.array([1, 0, 1]).reshape(1,-1), 
                                   np.array([2, 2, 2]).reshape(1,-1)), 4)
        self.assertEqual(k.compute(np.array((1, 0, 2)).reshape(1,-1), 
                                   np.array((-1, 2, 5)).reshape(1,-1)), 9)
                         
        self.assertAlmostEqual(k.compute(
                    np.array([1.2, -0.4, -2]).reshape(1,-1), 
                    np.array([4, 1.2, .5]).reshape(1,-1))[0], 3.32)
        self.assertAlmostEqual(k.compute(
                    np.array((1.2, -0.4, -2)).reshape(1,-1), 
                    np.array([4, 1.2, .5]).reshape(1,-1))[0], 3.32)

        with self.assertRaises(ValueError):
            k.compute(np.array([1, 0, 1]).reshape(1,-1), 
                      np.array([2, 2]).reshape(1,-1))
            
    def test_serialize(self):
        self._test_serialize(kernel.LinearKernel())
    
    def test_persists(self):
        self._test_persist(kernel.LinearKernel())
    
    def test_json(self):
        self._test_json(kernel.LinearKernel(),
                        {'class': 'LinearKernel'})


class TestPolynomialKernel(BaseTest, unittest.TestCase):
    def test_compute(self):
        with self.assertRaises(ValueError):
            kernel.PolynomialKernel(3.2)

        with self.assertRaises(ValueError):
            kernel.PolynomialKernel(-2)

        p = kernel.PolynomialKernel(2)
        self.assertEqual(p.compute(np.array((1, 0, 2)).reshape(1,-1), 
                                   np.array((-1, 2, 5)).reshape(1,-1)), 100)
        self.assertAlmostEqual(p.compute(
                    np.array([1.2, -0.4, -2]).reshape(1,-1), 
                    np.array([4, 1.2, .5]).reshape(1,-1)), 18.6624)

        p = kernel.PolynomialKernel(5)
        self.assertEqual(p.compute(np.array((1, 0, 2)).reshape(1,-1), 
                                   np.array([-1, 2, 5]).reshape(1,-1)), 10 ** 5)
        self.assertAlmostEqual(p.compute(
                    np.array((1.2, -0.4, -2)).reshape(1,-1), 
                    np.array((4, 1.2, .5)).reshape(1,-1)),
                1504.59195, delta=10**-6)

        with self.assertRaises(ValueError):
            p.compute(np.array((1, 0, 2)).reshape(1,-1), 
                      np.array((-1, 2)).reshape(1,-1))
    
    def test_serialize(self):
        for d in [2, 5]:
            self._test_serialize(kernel.PolynomialKernel(d))
    
    def test_persists(self):
        for d in [2, 5]:
            self._test_persist(kernel.PolynomialKernel(d))
    
    def test_json(self):
        for d in [2, 5]:
            self._test_json(kernel.PolynomialKernel(d),
                        {'class': 'PolynomialKernel', 'degree': d})



class TestHomogeneousPolynomialKernel(BaseTest, unittest.TestCase):
    def test_compute(self):
        with self.assertRaises(ValueError):
            kernel.HomogeneousPolynomialKernel(3.2)

        with self.assertRaises(ValueError):
            kernel.HomogeneousPolynomialKernel(-2)

        h = kernel.HomogeneousPolynomialKernel(2)
        self.assertEqual(h.compute(np.array((1, 0, 2)).reshape(1,-1), 
                                   np.array((-1, 2, 5)).reshape(1,-1)), 81.0)
        self.assertAlmostEqual(h.compute(
                    np.array([1.2, -0.4, -2]).reshape(1,-1), 
                    np.array([4, 1.2, .5]).reshape(1,-1))[0], 11.0224)

        h = kernel.HomogeneousPolynomialKernel(5)
        self.assertEqual(h.compute(
                    np.array((1, 0, 2)).reshape(1,-1), 
                    np.array([-1, 2, 5]).reshape(1,-1)) , 59049.0)
        self.assertAlmostEqual(h.compute(
                    np.array((1.2, -0.4, -2)).reshape(1,-1), 
                    np.array((4, 1.2, .5)).reshape(1,-1)),
                403.357761, delta=10**-6)

        with self.assertRaises(ValueError):
            h.compute(np.array((1, 0, 2)).reshape(1,-1), 
                      np.array((-1, 2)).reshape(1,-1))

    def test_serialize(self):
        for d in [2, 5]:
            self._test_serialize(kernel.HomogeneousPolynomialKernel(d))


class TestGaussianKernel(BaseTest, unittest.TestCase):
    def test_compute(self):
        with self.assertRaises(ValueError):
            kernel.GaussianKernel(-5)

        k = kernel.GaussianKernel(1)
        self.assertAlmostEqual(k.compute(np.array((1, 0, 1)).reshape(1,-1), 
                                         np.array((0, 0, 1)).reshape(1,-1))[0],
                               0.60653065)
        self.assertAlmostEqual(k.compute(
                    np.array([-3, 1, 0.5]).reshape(1,-1), 
                    np.array([1, 1.2, -8]).reshape(1,-1))[0], 6.73e-20)
        self.assertAlmostEqual(k.compute(
                    np.array([-1, -4, 3.5]).reshape(1,-1), 
                    np.array((1, 3.2, 6)).reshape(1,-1))[0], 3.29e-14)

        with self.assertRaises(ValueError):
            k.compute(np.array([-1, 3.5]).reshape(1,-1), 
                      np.array((1, 3.2, 6)).reshape(1,-1))
            
    def test_serialize(self):
        for sigma in [0.1, 1]:
            self._test_serialize(kernel.GaussianKernel(sigma))


class TestNormalizedLinearKernel(BaseTest, unittest.TestCase):
    def test_compute(self):
        with self.assertRaises(ValueError):
            kernel.NormalizedLinearKernel(0)

        k = kernel.NormalizedLinearKernel(1)
        self.assertAlmostEqual(k.compute(np.array((1, 0)).reshape(1, -1),
                                         np.array((0, 1)).reshape(1, -1))[0],
                               0.5)
        self.assertAlmostEqual(k.compute(np.array((1, 2)).reshape(1, -1),
                                         np.array((3, 4)).reshape(1, -1))[0],
                               12 / np.sqrt(6 * 26))

    def test_unit_diagonal(self):
        X = np.random.default_rng(0).normal(size=(20, 3))
        for offset in (0.01, 1, 10):
            k = kernel.NormalizedLinearKernel(offset)
            np.testing.assert_allclose(k.compute(X, X), 1)

    def test_broadcast(self):
        X = np.random.default_rng(0).normal(size=(5, 3))
        k = kernel.NormalizedLinearKernel(0.5)
        np.testing.assert_allclose(k.compute(X[0], X),
                                   [k.compute(X[[0]], X[[i]])[0]
                                    for i in range(5)])

    def test_positive_semidefinite(self):
        X = np.random.default_rng(0).normal(size=(30, 3))
        k = kernel.NormalizedLinearKernel(0.3)
        gram = np.array([k.compute(x, X) for x in X])
        self.assertGreater(np.linalg.eigvalsh(gram).min(), -1e-10)

    def test_serialize(self):
        for offset in [0.1, 1]:
            self._test_serialize(kernel.NormalizedLinearKernel(offset))


class TestHyperbolicKernel(BaseTest, unittest.TestCase):
    def test_compute(self):
        k = kernel.HyperbolicKernel(1, 5)
        self.assertAlmostEqual(k.compute(
                    np.array((1, 0, 1)).reshape(1,-1), 
                    np.array((0, 0, 1)).reshape(1,-1))[0], 0.9999877)
        self.assertAlmostEqual(k.compute(
                    np.array([-3, 1, 0.5]).reshape(1,-1), 
                    np.array([1, 1.2, -8]).reshape(1,-1))[0],
                -0.6640367, delta=10**-7)
        self.assertAlmostEqual(k.compute(
                    np.array([-1, -4, 3.5]).reshape(1,-1), 
                    np.array((1, 3.2, 6)).reshape(1,-1))[0],
                0.9999999, delta=10**-7)

        with self.assertRaises(ValueError):
            k.compute(np.array([-1, 3.5]).reshape(1,-1), 
                      np.array((1, 3.2, 6)).reshape(1,-1))
    
    def test_serialize(self):
        for alpha in [0.1, 1]:
            for beta in [0.1, 1]:
                self._test_serialize(kernel.HyperbolicKernel(alpha, beta))


class TestPrecomputedKernel(BaseTest, unittest.TestCase):
    def setUp(self):
        self.kernel = kernel.PrecomputedKernel(np.array(([1, 2], [3, 4])))

    def test_compute(self):
        with self.assertRaises(ValueError):
            kernel.PrecomputedKernel(np.array(([1, 2], [3, 4, 5])))

        self.assertEqual(self.kernel.compute(
                    np.array([1]).reshape(1,-1),
                    np.array([1]).reshape(1,-1)), [4.0])
        self.assertEqual(self.kernel.compute(
                    np.array([1]).reshape(1,-1),
                    np.array([0]).reshape(1,-1)), [3.0])

        with self.assertRaises(IndexError):
            self.kernel.compute(np.array([1]).reshape(1,-1),
                               np.array([2]).reshape(1,-1))

        with self.assertRaises(IndexError):
            self.kernel.compute(np.array([0]).reshape(1,-1),
                                np.array([1.6]).reshape(1,-1))
            
    def test_serialize(self):
        self._test_serialize(self.kernel)


if __name__ == '__main__':
    unittest.main()
