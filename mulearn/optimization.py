"""Implementation of optimization procedures.

This module module contains the implementations of the optimization processes
behind fuzzy inference.

Once loaded, the module preliminarily verifies that some libraries are
installed (notably, Gurobi and TensorFlow), emitting a warning otherwise.
Note that at least one of these libraries is needed in order to solve the
optimization problems involved in the fuzzy inference process.

"""

import numpy as np
import itertools as it
from collections.abc import Iterable
import logging

import copy

import mulearn.kernel as kernel

logger = logging.getLogger(__name__)

try:
    from gurobipy import LinExpr, GRB, Model, Env, QuadExpr, GurobiError

    gurobi_ok = True
except ModuleNotFoundError:
    # logger.warning('gurobi not available')
    gurobi_ok = False

try:
    import clarabel
    from scipy import sparse

    clarabel_ok = True
except ModuleNotFoundError:
    clarabel_ok = False

try:
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf

    tensorflow_ok = True
    from tensorflow.keras.optimizers import Adam

    logging.getLogger('tensorflow').setLevel(logging.ERROR)
except ModuleNotFoundError:
    # logger.warning('tensorflow not available')
    tensorflow_ok = False

try:
    import tqdm

    tqdm_ok = True
except ModuleNotFoundError:
    logger.warning('tqdm not available')
    tqdm_ok = False



def nearest_psd(k, rtol=1e-8):
    """Symmetrize a Gram matrix and remove its negative eigenvalues when
    they are due to rounding.

    Gram matrices of positive semidefinite kernels can have slightly negative
    eigenvalues because of rounding (e.g., for Gaussian kernels with large
    sigma). Solvers such as gurobi would then regard the problem as nonconvex
    and resort to a much slower global optimization. The spectrum is thus
    shifted when its negative part is below `rtol` times the largest
    eigenvalue; otherwise (e.g., for the hyperbolic kernel, which is not
    positive semidefinite) the matrix is only symmetrized.

    :returns: tuple -- the matrix and a flag telling whether it is positive
      semidefinite."""

    k = np.asarray(k, dtype=float)
    k = (k + k.T) / 2
    eigenvalues = np.linalg.eigvalsh(k)
    smallest, largest = eigenvalues[0], max(abs(eigenvalues[-1]), 1e-300)
    if smallest >= 0:
        return k, True
    if -smallest <= rtol * largest:
        return k + (-smallest + 1e-12 * largest) * np.eye(len(k)), True
    return k, False


class Solver:
    """Abstract solver for optimization problems.

    The base class for solvers is :class:`Solver`: it exposes a method
    `solve` which delegates the numerical optimization process to an abstract
    method `solve_problem` and subsequently clips the results to the boundaries
    of the feasible region.
    """

    def solve_problem(self, *args):
        pass

    def solve(self, xs, mus, c, k):
        """Solve optimization phase.

        Build and solve the constrained optimization problem on the basis
        of the fuzzy learning procedure.

        :param xs: Objects in training set.
        :type xs: iterable
        :param mus: Membership values for the objects in `xs`.
        :type mus: iterable
        :param c: constant managing the trade-off in joint radius/error
          optimization.
        :type c: float
        :param k: Kernel function to be used.
        :type k: :class:`mulearn.kernel.Kernel`
        :raises: ValueError if c is non-positive or if xs and mus have
          different lengths.
        :returns: `list` -- optimal values for the independent variables
          of the problem."""
        if c <= 0:
            raise ValueError('c should be positive')
        
        mus = np.array(mus)
        chis = self.solve_problem(xs, mus, c, k)

        chis_opt = [np.clip(ch, l, u)
                    for ch, l, u in zip(chis, -c * (1 - mus), c * mus)] # noqa

        return chis_opt
    
    def __eq__(self, other):
        """Check solver equality w.r.t. other objects."""
        return type(self) is type(other) and self.__dict__ == other.__dict__


class GurobiSolver(Solver):
    """Solver based on gurobi.

    Using this class requires that gurobi is installed and activated
    with a software key. The library is available at no cost for academic
    purposes (see
    https://www.gurobi.com/downloads/end-user-license-agreement-academic/).
    Alongside the library, also its interface to python should be installed,
    via the gurobipy package.
    """

    default_values = {"time_limit": 10 * 60,
                      "adjustment": 0,
                      "initial_values": None,
                      "threads": 0}

    def __init__(self, time_limit=default_values['time_limit'],
                 adjustment=default_values['adjustment'],
                 initial_values=default_values['initial_values'],
                 threads=default_values['threads']):
        """
        Build an object of type GurobiSolver.

        :param time_limit: Maximum time (in seconds) before stopping iterative
          optimization, defaults to 10*60.
        :type time_limit: int
        :param adjustment: Adjustment value to be used with non-PSD matrices,
          defaults to 0. Specifying `'auto'` instead than a numeric value
          will automatically trigger the optimal adjustment if needed.
        :type adjustment: float or `'auto'`
        :param initial_values: Initial values for variables of the optimization
          problem, defaults to None.
        :type initial_values: iterable of floats or None
        :param threads: Number of threads used by gurobi, defaults to 0
          (automatic choice). Set it to 1 when several problems are solved
          in parallel processes.
        :type threads: int
        """
        self.time_limit = time_limit
        self.adjustment = adjustment
        self.initial_values = initial_values
        self.threads = threads


    def solve_problem(self, xs, mus, c, k):
            """Optimize via gurobi.
    
            Build and solve the constrained optimization problem at the basis
            of the fuzzy learning procedure using the gurobi API.
    
            :param xs: objects in training set.
            :type xs: iterable
            :param mus: membership values for the objects in `xs`.
            :type mus: iterable
            :param c: constant managing the trade-off in joint radius/error
              optimization.
            :type c: float
            :param k: kernel computations to be used.
            :type k: iterable
            :raises: ValueError if optimization fails or if gurobi is not installed
            :returns: list -- optimal values for the independent variables of the
              problem.
            """

            if not gurobi_ok:
                raise ValueError('gurobi not available')
    
            m = len(xs)
            k, _ = nearest_psd(k)

            with Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                with Model('mulearn', env=env) as model:
                    model.setParam('OutputFlag', 0)
                    model.setParam('TimeLimit', self.time_limit)
                    model.setParam('Threads', self.threads)
                    #model.setParam('NonConvex', 2)

                    if c < np.inf:
                        model.addVars(m,
                                     name=[f'chi_{i}' for i in range(m)],
                                     lb=-c * (1 - mus), ub=c * mus,
                                     vtype=GRB.CONTINUOUS)
                    else:
                        model.addVars(name=[f'chi_{i}' for i in range(m)], vtype=GRB.CONTINUOUS)
    
                    model.update()
                    chis = np.array(model.getVars())

                    if self.initial_values is not None:
                        for c, i in zip(chis, self.initial_values):
                            c.start = i
                
                    obj = QuadExpr()

                    obj.add(chis.dot(k.dot(chis)))

                    obj.add(chis.dot(-1*np.diag(k)))
    
                    if self.adjustment and self.adjustment != 'auto':
                        obj.add(self.adjustment * chis.dot(chis))
    
                    model.setObjective(obj, GRB.MINIMIZE)
    
                    constEqual = LinExpr()
                    constEqual.add(sum(chis), 1.0)
    
                    model.addConstr(constEqual == 1)
    
                    try:
                        model.optimize()
                    except GurobiError as e:
                        print(e.message)
                        if self.adjustment == 'auto':
                            s = e.message
                            a = float(s[s.find(' of ') + 4:s.find(' would')])
                            logger.warning('non-diagonal Gram matrix, '
                                           f'retrying with adjustment {a}')

                            obj.add(a * chis.dot(chis))
                            model.setObjective(obj, GRB.MINIMIZE)
    
                            model.optimize()
                        else:
                            raise e
    
                    if model.Status != GRB.OPTIMAL:

                        if model.Status == GRB.ITERATION_LIMIT:
                            logger.warning('gurobi: optimization terminated because the total number of simplex \
                            iterations performed exceeded the value specified in the IterationLimitparameter, \
                            or because the total number of barrier iterations exceeded the value specified in the BarIterLimit parameter.')

                        elif model.Status == GRB.NODE_LIMIT:
                            logger.warning('gurobi: optimization terminated because the total number of \
                            branch-and-cut nodes explored exceeded the value specified in the NodeLimit parameter.')
                            
                        elif model.Status == GRB.TIME_LIMIT:
                            logger.warning('gurobi: optimization terminated because the time expended \
                            exceeded the value specified in the TimeLimit parameter.')
                            
                        elif model.Status == GRB.SUBOPTIMAL:
                            logger.warning('gurobi: optimization terminated with a sub-optimal solution!')

                        else:
                            logger.warning(f'gurobi: optimal solution not found! ERROR CODE: {model.Status}')
                            
    
                    return [ch.x for ch in chis]

    
    def __repr__(self):
        args = []

        for a in self.default_values:
            if self.__getattribute__(a) != self.default_values[a]:
                args.append(f"{a}={self.__getattribute__(a)}")
        return f"{self.__class__.__name__}({', '.join(args)})"




class ClarabelSolver(Solver):
    """Solver based on Clarabel, an open source interior point solver.

    Clarabel (https://clarabel.org, Apache 2.0 license) is installed via the
    clarabel package and does not require any license key.
    """

    default_values = {"tol": 1e-8, "max_iter": 200}

    def __init__(self, tol=default_values['tol'],
                 max_iter=default_values['max_iter']):
        """Build an object of type ClarabelSolver.

        :param tol: Tolerance on the duality gap and on the feasibility of
          the solution, defaults to 1e-8.
        :type tol: float
        :param max_iter: Maximum number of interior point iterations,
          defaults to 200.
        :type max_iter: int
        """
        self.tol = tol
        self.max_iter = max_iter

    def solve_problem(self, xs, mus, c, k):
        """Minimize $\\chi^\\top K \\chi - \\mathrm{diag}(K)^\\top \\chi$
        subject to $\\sum_i \\chi_i = 1$ and
        $-c(1 - \\mu_i) \\leq \\chi_i \\leq c \\mu_i$.

        :raises: ValueError if clarabel is not available, RuntimeError if the
          problem could not be solved."""

        if not clarabel_ok:
            raise ValueError('clarabel not available')

        n = len(k)
        k_psd, is_psd = nearest_psd(k)
        if not is_psd:
            raise ValueError('the Gram matrix is not positive semidefinite '
                             '(use GurobiSolver for nonconvex problems)')
        P = sparse.csc_matrix(np.triu(2 * k_psd))
        q = -np.diag(k).astype(float)
        # equality constraint, then upper and lower bounds as  A x + s = b,
        # with s in the zero cone and in the nonnegative orthant
        A = sparse.vstack([sparse.csc_matrix(np.ones((1, n))),
                           sparse.eye(n), -sparse.eye(n)]).tocsc()
        b = np.concatenate([[1], c * mus, c * (1 - mus)])
        cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(2 * n)]

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = self.max_iter
        settings.tol_gap_abs = settings.tol_gap_rel = self.tol
        settings.tol_feas = self.tol

        solution = clarabel.DefaultSolver(P, q, A, b, cones, settings).solve()
        if solution.status not in (clarabel.SolverStatus.Solved,
                                   clarabel.SolverStatus.AlmostSolved):
            raise RuntimeError(f'clarabel: {solution.status}')
        return np.array(solution.x)

    def __repr__(self):
        args = [f"{a}={getattr(self, a)}" for a in self.default_values
                if getattr(self, a) != self.default_values[a]]
        return f"{self.__class__.__name__}({', '.join(args)})"


class TensorFlowSolver(Solver):
    """Solver based on TensorFlow.

    Using this class requires that TensorFlow 2.X is installed."""

    default_values = {"initial_values": "random",
                      "init_bound": 0.1,
                      "n_iter": 100,
                      "optimizer": Adam(learning_rate=1e-3)
                                   if tensorflow_ok else None,  # noqa
                      "tracker": tqdm.trange if tqdm_ok else range}

    def __init__(self,
                 initial_values=default_values["initial_values"],
                 init_bound=default_values["init_bound"],
                 n_iter=default_values["n_iter"],
                 optimizer=default_values["optimizer"],
                 tracker=default_values["tracker"]):
        """Build an object of type TensorFlowSolver.

        :param initial_values: values to be used for initializing the
          independent  variables, either randomly (if set to `'random'`)` or
          seting them to a given sequence of initial values (if set to an
          iterable of floats), defaults to `'random'`.
        :type initial_values: `str` or iterable of `float`
        :param init_bound: Absolute value of the extremes of the interval used
          for random initialization of independent variables, defaults to 0.1.
        :type init_bound: `float`
        :param n_iter: Number of iterations of the optimization process,
          defaults to 100.
        :type n_iter: `int`
        :param optimizer: Optimization algorithm to be used, defaults to Adam
          with learning rate=1e-4 if tensorflow is available, to None otherwise.
        :type optimizer: :class:`tf.keras.optimizers.Optimizer`
        :param tracker: Tool to graphically depict the optimization progress,
          defaults to `tqdm.trange` if tqdm is available, to `range` (i.e., no
          graphical tool) otherwise.
        :type tracker: `object`
        """

        self.init_bound = init_bound
        self.initial_values = initial_values
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.tracker = tracker

    def solve_lagrange_relaxation(self, Q, q, A, b, C, d,
                  max_iter=10,
                  max_gap=10**-2,
                  alpha_0 = 10**-3,
                  window_width = 10,
                  verbose=False):
        """Solves the lagrangian relaxation for a constrained optimization
        problem and returns its result. The structure of the primal problem
        is the following
        
        min x.T Q x + q.T x
        subject to
        A x = b
        C x <= d
        
        where .T denotes the transposition operator. Optimization takes place
        in a iterated two-steps procedure: an outer process devoted to modifying
        the values of the lagrange multipliers, and an inner process working on
        the primal variables.
        
        The arguments are as follows, given n as the number of variables of
        the primal problem (i.e., the length of x)
        
        - Q: n x n matrix containing the quadratic coefficients of the cost
        function;
        - q: vector containing the n linear coefficients of the cost function
        - A: s x n matrix containing the coefficients of the = constraints
        - b: vector containing the s right members of the = constraints
        - C: t x n matrix containing the coefficients of the <= constraings
        - d: vector containing the t right members of the <= coefficients
        - max_iter: maximum number of iterations of the *outer* optimization
        procedure
        - max_gap: maximum gap between primal and dual objectives ensuring
        premature end of the *outer* optimization procedure
        - alpha_0: initial value of the learning rate in the *outer* optimization
        procedure
        - window_width: width of the moving window on the objective function for
        the *inner* optimization process
        - verbose: boolean flag triggering verbose output
        
        returns
        """

        #TODO add possibility to specify initial values

        x = tf.Variable(np.random.random(len(q)),
                        name='x',
                        trainable=True,
                        dtype=tf.float32)
        Q = tf.constant(np.array(Q), dtype='float32')
        q = tf.constant(np.array(q), dtype='float32')
        
        A = np.array(A)
        s = len(A)
        C = np.array(C)
        b = np.array(b)
        d = np.array(d)
        
        M = np.vstack([A, -A, C])
        m = np.hstack([b, -b, d])
        lambda_ = tf.constant(np.random.random(len(m)), dtype='float32')
        
        M = tf.constant(M, dtype='float32')
        m = tf.constant(m, dtype='float32')

        
        def original_objective():
            def obj():
                return tf.tensordot(tf.linalg.matvec(Q, x), x, axes=1) + \
                    tf.tensordot(q, x, axes=1)
            return obj

        def lagrangian_objective(lambda_):
            def obj():
                return tf.tensordot(tf.linalg.matvec(Q, x), x, axes=1) + \
                    tf.tensordot(q, x, axes=1) + \
                    tf.tensordot(lambda_, m - tf.linalg.matvec(M, x), axes=1)
            return obj

        obj_val = []
        lagr_val = []
        gap_val = []
        gap = max_gap + 1

        num_bad_iterations = 0
        prev_orig = np.inf

        i = 0
        
        while i < max_iter and (gap<0 or gap > max_gap):
            lagr_obj = lagrangian_objective(lambda_)
            orig_obj = original_objective()
            prev_lagr = 10*3
            curr_lagr = 0
            vals = []
            t = 0
            window_width = 30
            window = list(np.logspace(1, window_width, window_width))
            # this is to ensure a high value for the standard deviation
            # of the elements to which the window has been initialized 
            
            while (np.std(window)/abs(np.mean(window)) > 0.001 or t < 100) \
                and t < 1000:
                self.optimizer.minimize(lagr_obj, var_list=x)
                prev_lagr = curr_lagr
                curr_lagr = lagr_obj().numpy()
                vals.append(curr_lagr)
                t += 1
                window = window[1:]
                window.append(curr_lagr)
            
            curr_orig = orig_obj().numpy()
            if curr_orig < prev_orig:
                num_bad_iterations += 1

            prev_orig = curr_orig

            
            obj_val.append(curr_orig)
            lagr_val.append(curr_lagr)
            
            subgradient = (m - tf.linalg.matvec(M, x)).numpy()
            gap = tf.tensordot(lambda_[:2*s], m[:2*s] - \
                tf.linalg.matvec(M[:2*s], x), axes=1).numpy()
            gap_val.append(gap)
            
            if verbose and i%1 == 0:
                print(f'i={i}, dual={lagr_obj().numpy():.3f}, '
                    f'prim={orig_obj().numpy():.3f}, '
                    f'gap={gap:.6f}')
            
            alpha = alpha_0 / num_bad_iterations
            lambda_ = tf.maximum(0, lambda_ + alpha * subgradient)

            i += 1

        return obj_val, lagr_val, x, lambda_, gap_val

    def solve_problem(self, xs, mus, c, k):
        if not tensorflow_ok:
            raise ValueError('tensorflow not available')

        m = len(xs)

        if self.initial_values == 'random':
            alphas = [tf.Variable(ch, name=f'alpha_{i}',
                                  trainable=True, dtype=tf.float32)
                      for i, ch in enumerate(np.random.uniform( \
                                -self.init_bound, self.init_bound, m))]
            betas = [tf.Variable(ch, name=f'beta_{i}',
                                  trainable=True, dtype=tf.float32)
                      for i, ch in enumerate(np.random.uniform( \
                                -self.init_bound, self.init_bound, m))]

        elif isinstance(self.initial_values, Iterable):
            alphas = [tf.Variable(ch, name=f'alpha_{i}',
                                trainable=True, dtype=tf.float32)
                    for i, ch in enumerate(self.initial_values)]
            betas = [tf.Variable(ch, name=f'beta_{i}',
                                trainable=True, dtype=tf.float32)
                    for i, ch in enumerate(self.initial_values)]

        else:
            raise ValueError("`initial_values` should either be set to "
                             "'random' or to a list of initial values.")
        
        x = alphas + betas

        K11 = np.array([[-mu_i * mu_j for mu_j in mus] for mu_i in mus]) * k
        K00 = np.array([[-(1-mu_i) * (1 - mu_j) for mu_j in mus]
                         for mu_i in mus]) * k
        K01 = np.array([[2 * mu_i * (1-mu_j) for mu_j in mus]
                         for mu_i in mus]) * k
        Z = np.zeros((m, m))

        Q = -np.vstack((np.hstack((K11, K01)), np.hstack((Z, K00))))
        q = -np.hstack((np.diag(k) * mus, np.diag(k) * (1 - mus)))

        A = np.array([np.hstack((mus, 1-mus))])
        b = np.array([1])

        C = np.vstack((np.hstack((np.identity(m), np.zeros((m, m)))),
                       np.hstack((np.zeros((m, m)), -np.identity(m)))))
        d = np.hstack((np.zeros(m), - c * np.ones(m)))

        x = np.random.random(2*m)
        assert(Q.shape == (2*m, 2*m))
        assert(len(q) == 2*m)
        assert(type(x @ (Q @ x) + q @ x) is np.float64)

        assert(A.shape == (1, 2*m))
        assert(len(b) == 1)
        assert(type(A@x - b) == np.ndarray)
        assert(len(A@x - b) == 1)

        assert(C.shape == (2*m, 2*m))
        assert(len(d) == 2*m)
        assert(type(C @ x - d) == np.ndarray)
        assert(len(C @ x - d) == 2*m)

        _, _, x, _, _ = self.solve_lagrange_relaxation(Q, q, A, b, C, d, verbose=True)

        alphas = np.array(x[:m])
        betas = np.array(x[m:])

        return alphas * np.array(mus) - betas * (1 - np.array(mus))


    def __repr__(self):
        obj_repr = f"TensorFlowSolver("

        for a in ("initial_values", "init_bound", "n_iter",
                  "optimizer", "tracker"):
            if self.__getattribute__(a) != self.default_values[a]:
                obj_repr += f", {a}={self.default_values[a]}"
        return obj_repr + ")"




