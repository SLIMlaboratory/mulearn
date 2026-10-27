import copy
import itertools as it
import logging
import multiprocessing as mp
import pickle
import time
import unittest
from unittest import mock
import warnings

from joblib import Parallel, delayed
import numpy as np
from sklearn.datasets import load_iris
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.base import is_regressor
from sklearn.exceptions import NotFittedError

from mulearn import FuzzyInductor
from mulearn.fuzzifier import CrispFuzzifier, ExponentialFuzzifier
from mulearn.kernel import LinearKernel, PolynomialKernel, GaussianKernel
from mulearn.kernel import HyperbolicKernel

RANDOM_STATE = 42
NUM_CORES = mp.cpu_count()

def make_hp_configurations(grid):
    return [{n: v for n, v in zip(grid.keys(), t)}
            for t in it.product(*grid.values())]

def fit_and_score(estimator,
                  X_trainval, y_trainval,
                  hp_configuration, model_selection,
                  scorer=metrics.root_mean_squared_error):

    estimator.set_params(**hp_configuration)
    current_scores = []
    for train_index, val_index in model_selection.split(X_trainval, y_trainval):
        X_train, X_val = X_trainval[train_index], X_trainval[val_index]
        y_train, y_val = y_trainval[train_index], y_trainval[val_index]

        estimator.fit(X_train, y_train)
        y_hat = estimator.predict(X_val)
        score = scorer(y_val, y_hat)
        current_scores.append(score)

    return np.mean(current_scores), hp_configuration

def learn_parallel(X, y, estimator, param_grid,
                   model_selection=StratifiedKFold(n_splits=5,
                                                   shuffle=True,
                                                   random_state=RANDOM_STATE),
                   model_assessment=StratifiedKFold(n_splits=5,
                                                    shuffle=True,
                                                    random_state=RANDOM_STATE),
                   gs_scorer=metrics.root_mean_squared_error,
                   test_scorers=[metrics.root_mean_squared_error,
                                 metrics.hinge_loss],
                   test_scorer_names=['RMSE', 'Hinge'],
                   n_jobs=-1, pre_dispatch=None):

    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    ping = time.time()

    outer_scores = []

    for trainval_index, test_index in model_assessment.split(X, y):
        X_trainval, X_test = X[trainval_index], X[test_index]
        y_trainval, y_test = y[trainval_index], y[test_index]

        gs_result = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)( \
                    delayed(fit_and_score)(copy.deepcopy(estimator),
                                           X_trainval, y_trainval,
                                           hp_conf,
                                           model_selection=model_selection,
                                           scorer=gs_scorer)
                            for hp_conf in make_hp_configurations(param_grid))

        best_conf = sorted(gs_result, key=lambda t: t[0])[0][1]
        estimator.set_params(**best_conf)
        estimator.fit(X_trainval, y_trainval)

        y_hat = estimator.predict(X_test)
        outer_scores.append([score(y_test, y_hat) for score in test_scorers])

    pong = time.time()
    # Refit estimator with best configuration
    # of last external cv fold on all data
    estimator.fit(X, y)

    avg = np.mean(outer_scores, axis=0)
    std = np.std(outer_scores, axis=0, ddof=1)
    result = {'model': estimator.__class__.__name__, 'type': 'FINAL'} | \
             {n + ' mean': m for n, m in zip(test_scorer_names, avg)} | \
             {n + ' std': s for n, s in zip(test_scorer_names, std)} | \
             {'time': pong-ping}

    return estimator, best_conf, result


class TestFuzzyInductor(unittest.TestCase):
    def setUp(self):
        d = load_iris()
        self.X = d['data']
        y = d['target']
        y[y==2] = 0
        self.y = y

    def test_is_regressor(self):
        self.assertTrue(is_regressor(FuzzyInductor()))

    def test_in_pipeline(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        fi = FuzzyInductor()
        self.assertNotEqual(fi, 'passthrough')
        self.assertEqual(fi, FuzzyInductor())
        model = make_pipeline(StandardScaler(), fi).fit(self.X, self.y)
        self.assertEqual(len(model.predict(self.X)), len(self.X))

    def test_not_fitted(self):
        with self.assertRaises(NotFittedError):
            FuzzyInductor().predict(self.X)

    def test_infeasible(self):
        y = np.zeros(len(self.y))
        y[:5] = 1
        with self.assertRaisesRegex(ValueError, 'infeasible'):
            FuzzyInductor(c=.1).fit(self.X, y)

    def test_feasibility_bound(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            y = rng.uniform(0, 1, len(self.X))
            fi = FuzzyInductor(c=1 / np.sum(y), k=GaussianKernel(1)).fit(self.X, y)
            np.testing.assert_allclose(fi.chis_, y / np.sum(y), atol=1e-6)

    def test_score(self):
        fi = FuzzyInductor().fit(self.X, self.y)
        mse = np.mean((fi.predict(self.X) - self.y) ** 2)
        self.assertAlmostEqual(fi.score(self.X, self.y), -mse)

    def test_pinball_radius(self):
        fi = FuzzyInductor(c=.5)
        sq_dist = np.array([1., 2, 3, 4])
        y = np.array([1, 1, 0, 0])
        objective = lambda r: r + .5 * sum(
            m * max(d - r, 0) + (1 - m) * max(r - d, 0)
            for d, m in zip(sq_dist, y))
        grid = np.linspace(0, 5, 5001)
        self.assertAlmostEqual(objective(fi._pinball_sq_radius(sq_dist, y)),
                               min(map(objective, grid)))

    def test_sq_radius(self):
        # numerical solvers return dual variables only approximately equal
        # to the bounds: the squared radius must nonetheless be the squared
        # distance of the unbounded support vectors, which lie on the sphere
        from mulearn.optimization import ClarabelSolver, GurobiSolver
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 2))
        mu = np.clip(rng.uniform(-.3, 1.3, 40), 0, 1)
        for solver in (ClarabelSolver(), GurobiSolver()):
            fi = FuzzyInductor(c=1, solver=solver).fit(X, mu)
            chis = np.array(fi.chis_)
            sq_dist = fi.x_to_sq_dist(X)
            free = (chis > -(1 - mu) + 1e-4) & (chis < mu - 1e-4)
            self.assertGreater(free.sum(), 0)
            np.testing.assert_allclose(sq_dist[free],
                                       fi.fuzzifier.sq_radius_05, atol=1e-5)

    def test_no_support_vectors(self):
        X = np.array([[0.], [1.], [2.], [3.]])
        y = np.array([1, 1, 0, 0])
        fi = FuzzyInductor(c=.5, k=GaussianKernel(.5),
                           fuzzifier=CrispFuzzifier())
        with mock.patch.object(fi, '_pinball_sq_radius',
                               wraps=fi._pinball_sq_radius) as pinball, \
             mock.patch('mulearn.optimization.GurobiSolver.solve',
                        return_value=[.5, .5, 0., 0.]):
            fi.fit(X, y)
        pinball.assert_called_once()
        self.assertTrue(np.all((fi.predict(X) >= 0) & (fi.predict(X) <= 1)))

    def test_serialization(self):
        fi = FuzzyInductor()
        fi.fit(self.X, self.y)
        s = pickle.dumps(fi)
        fi_clone = pickle.loads(s)

        self.assertEqual(fi, fi_clone)

    def test_fit(self):
        kernel = [LinearKernel(), PolynomialKernel(2),
                  GaussianKernel(.1), HyperbolicKernel()]
        # with the narrow Gaussian kernel all positive examples are
        # unbounded support vectors, thus lying on the learnt sphere, where
        # the fixed profile assigns membership 1/2: hence RMSE = sqrt(1/12)
        scores = [0.18015338625478378, 0.19079606373771996,
                  0.2886751345948285, 0.501304679791567]
        # the hyperbolic Gram matrix is almost rank one, so that the optimal
        # solution is not unique
        deltas = [1e-7, 1e-7, 1e-7, 1e-2]
        logging.disable(logging.WARNING)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for k, s, delta in zip(kernel, scores, deltas):
                fi = FuzzyInductor(k=k)
                fi.fit(self.X, self.y)
                y_hat = fi.predict(self.X)
                rmse = metrics.root_mean_squared_error(self.y, y_hat)
                self.assertAlmostEqual(s, rmse, delta=delta)
        logging.disable(logging.NOTSET)
    
    def test_standard_train(self):
        model = FuzzyInductor(fuzzifier=ExponentialFuzzifier(profile='fixed'))

        grid = {'c': np.linspace(0.1, 0.2, 2),
                'k': [GaussianKernel(.01), GaussianKernel(.1)]}
        cv_out = StratifiedKFold(n_splits=5, shuffle=True,
                                 random_state=RANDOM_STATE)
        cv_in = StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=RANDOM_STATE)
        gs = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error',
                          cv=cv_in, n_jobs=NUM_CORES, pre_dispatch=2*NUM_CORES)
        score = cross_val_score(gs, self.X, self.y,
                                scoring='neg_root_mean_squared_error',
                                cv=cv_out)
        # with these narrow kernels, test points are far from all training
        # points and get null membership: RMSE = sqrt(1/3) on each fold
        target = np.full(5, -1 / np.sqrt(3))

        for t, s in zip(score, target):
            self.assertAlmostEqual(t, s)
    
    def test_custom_train(self):
        model = FuzzyInductor()

        grid = {'c': np.linspace(0.1, 0.3, 2),
                'k': [GaussianKernel(.1), GaussianKernel(.01)]}

        n_cores = mp.cpu_count()
        model, best_conf, result = learn_parallel(self.X, self.y, model, grid,
                        n_jobs=NUM_CORES, pre_dispatch=2*NUM_CORES)

        result = {'configuration': best_conf} | result
        self.assertAlmostEqual(result['RMSE mean'], 1 / np.sqrt(3),
                               delta=1E-5)
        self.assertAlmostEqual(result['RMSE std'], 0, delta=1E-4)

if __name__ == '__main__':
    unittest.main()