"""Kernel implementation.

This module implements the kernel used in mulearn.
"""

from itertools import zip_longest

import json_fix
import numpy as np


def to_list(arg):
    return arg.tolist()  if isinstance(arg, np.ndarray) else arg


class Kernel:
    """Base kernel class."""

    def __init__(self):
        """Create an instance of :class:`Kernel`."""
        pass

    def compute(self, arg_1, arg_2):
        """Compute the kernel value, given two arrays of arguments.

        :param arg_1: First kernel array argument.
        :type arg_1: Object
        :param arg_2: Second kernel array argument.
        :type arg_2: Object
        :raises: NotImplementedError (:class:`Kernel` is abstract)
        :returns: `array` -- kernel values.
        """
        raise NotImplementedError(
            'The base class does not implement the `compute` method')

    def __str__(self):
        """Return the string representation of a kernel."""
        return self.__repr__()

    def __eq__(self, other):
        """Check kernel equality w.r.t. other objects."""
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Check kernel inequality w.r.t. other objects."""
        return not self == other

    @staticmethod
    def __nonzero__():
        """Check if a kernel is non-null."""
        return True

    def __hash__(self):
        """Generate hashcode for a kernel."""
        return hash(self.__repr__())

    @classmethod
    def get_default(cls):
        """Return the default kernel.

        :returns: `LinearKernel()` -- the default kernel.
        """
        return LinearKernel()
    
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __json__(self):
        return {'class': self.__class__.__name__} | self.__dict__


class LinearKernel(Kernel):
    """Linear kernel class."""

    def compute(self, arg_1, arg_2):
        r"""
        Compute the kernel value over several arguments.

        The value $k(x_1, x_2)$ of a linear kernel is equal to the dot product
        $x_1 \cdot x_2$, that is to $\sum_{i=1}^n (x_1)_i (x_2)_i$, $n$ being
        the common dimension of $x_1$ and $x_2$. Given the two arrays of kernels
        $Y$ and $Z$, the return value will be $(k(y_1,z_1), k(y_2,z_2),..., 
        k(y_m, z_m))$, $m$ being the number of elements in the arrays. 

        :param arg_1: First kernel array argument.
        :type arg_1: Object convertible to np.array
        :param arg_2: Second kernel array argument.
        :type arg_2: Object convertible to np.array
        :returns: `array` -- kernel values.
        """
        return np.sum(np.array(arg_1) * np.array(arg_2), axis=1)

    def __repr__(self):
        """Return the python representation of the kernel."""
        return 'LinearKernel()'


class PolynomialKernel(Kernel):
    """Polynomial kernel class."""

    def __init__(self, degree):
        r"""Create an instance of `PolynomialKernel`.

        :param degree: degree of the polynomial kernel.
        :type degree: `int`
        :raises: ValueError if `degree` is not an integer or if it has a
          negative value.
        """
        super().__init__()
        if degree > 0 and isinstance(degree, int):
            self.degree = degree
        else:
            raise ValueError(f"{degree} is not usable as a polynomial degree")

    def compute(self, arg_1, arg_2):
        r"""
        Compute the kernel value.

        The value $k(x_1, x_2)$ of a polynomial kernel is equal to the
        quantity $(x_1 \cdot x_2 + 1)^d$, $d$ being the polynomial degree of
        the kernel. Given the two arrays of kernels $Y$ and $Z$, the return 
        value will be $(k(y_1,z_1), k(y_2,z_2),..., k(y_m, z_m))$, $m$ being
        the number of elements in the arrays. 

        :param arg_1: First kernel array argument.
        :type arg_1: Object convertible to np.array
        :param arg_2: Second kernel array argument.
        :type arg_2: Object convertible to np.array
        :returns: `array` -- kernel values.
        """
        return (np.sum(np.array(arg_1) * np.array(arg_2),
                       axis=1) + 1) ** self.degree
    
    def __eq__(self, other):
        """Check kernel equality w.r.t. other objects."""
        return type(self) == type(other) and self.degree == other.degree
    
    def __repr__(self):
        """Return the python representation of the kernel."""
        return f"PolynomialKernel({self.degree})"


class HomogeneousPolynomialKernel(PolynomialKernel):
    """Homogeneous polynomial kernel class."""

    def __init__(self, degree):
        r"""Create an instance of `HomogeneousPolynomialKernel`.

        :param degree: degree of the polynomial kernel.
        :type degree: `int`
        :raises: ValueError if `degree` is not an integer or if it has a
          negative value.
        """
        super().__init__(degree)

    def compute(self, arg_1, arg_2):
        r"""Compute the kernel value.

        The value $k(x_1, x_2)$ of a homogeneous polynomial kernel is
        intended as the quantity $(x_1 \cdot x_2)^d$, $d$ being the polynomial
        degree of the kernel. Given the two arrays of kernels $Y$ and $Z$, 
        the return value will be $(k(y_1,z_1), k(y_2,z_2),..., k(y_m, z_m))$, 
        $m$ being the number of elements in the arrays. 

        :param arg_1: First kernel array argument.
        :type arg_1: Object convertible to np.array
        :param arg_2: Second kernel array argument.
        :type arg_2: Object convertible to np.array
        :returns: `array` -- kernel values.
        """
        return np.sum(np.array(arg_1) * np.array(arg_2),
                      axis=1) ** self.degree

    def __repr__(self):
        """Return the python representation of the kernel."""
        return f"HomogeneousPolynomialKernel({self.degree})"


class GaussianKernel(Kernel):
    """Gaussian kernel class."""
    
    default_sigma = 1

    def __init__(self, sigma=default_sigma):
        r"""Create an instance of `GaussianKernel`.

        :param sigma: gaussian standard deviation, defaults to 1.
        :type sigma: `float`
        :raises: ValueError if `sigma` has a negative value.
        """
        super().__init__()
        if sigma > 0:
            self.sigma = sigma
        else:
            raise ValueError(f'{sigma} is not usable '
                             'as a gaussian standard deviation')


    def compute(self, arg_1, arg_2):
        r"""Compute the kernel value.

        The value $k(x_1, x_2)$ of a gaussian kernel is intended as the
        quantity $\mathrm e^{-\frac{||x_1 - x_2||^2}{2 \sigma^2}}$, $\sigma$
        being the kernel standard deviation. Given the two arrays of kernels
        $Y$ and $Z$, the return value will be $(k(y_1,z_1), k(y_2,z_2),..., 
        k(y_m, z_m))$, $m$ being the number of elements in the arrays. 

        :param arg_1: First kernel array argument.
        :type arg_1: Object convertible to np.array
        :param arg_2: Second kernel array argument.
        :type arg_2: Object convertible to np.array
        :returns: `array` -- kernel values.
        """
        diff = np.linalg.norm(np.array(arg_1) - np.array(arg_2), axis=1) ** 2
        return np.exp(-1. * diff / (2 * self.sigma ** 2))

    def __repr__(self):
        """Return the python representation of the kernel."""
        obj_repr = "GaussianKernel("
        if self.sigma != self.default_sigma:
            obj_repr += f"sigma={self.sigma}"
        obj_repr += ")"
        return obj_repr


class NormalizedLinearKernel(Kernel):
    """Normalized linear kernel class."""

    default_offset = 1

    def __init__(self, offset=default_offset):
        r"""Create an instance of `NormalizedLinearKernel`.

        :param offset: value added to the components of the arguments in an
          additional dimension, defaults to 1.
        :type offset: `float`
        :raises: ValueError if `offset` has a negative value.
        """
        super().__init__()
        if offset > 0:
            self.offset = offset
        else:
            raise ValueError(f'{offset} is not usable as an offset')

    def compute(self, arg_1, arg_2):
        r"""Compute the kernel value.

        The value $k(x_1, x_2)$ of a normalized linear kernel is intended as
        the quantity $\frac{x_1 \cdot x_2 + s^2}{\sqrt{(||x_1||^2 + s^2)
        (||x_2||^2 + s^2)}}$, $s$ being the kernel offset. Therefore,
        $k(x, x) = 1$ for each $x$, so that the squared distance from the
        center of the sphere is an affine function of the image of $x$, whose
        level sets are the intersections of a hyperplane with the unit sphere.
        Given the two arrays of kernels $Y$ and $Z$, the return value will be
        $(k(y_1,z_1), k(y_2,z_2),..., k(y_m, z_m))$, $m$ being the number of
        elements in the arrays.

        :param arg_1: First kernel array argument.
        :type arg_1: Object convertible to np.array
        :param arg_2: Second kernel array argument.
        :type arg_2: Object convertible to np.array
        :returns: `array` -- kernel values.
        """
        arg_1, arg_2 = np.atleast_2d(arg_1), np.atleast_2d(arg_2)
        s_2 = self.offset ** 2
        return (np.sum(arg_1 * arg_2, axis=1) + s_2) / \
            np.sqrt((np.sum(arg_1 ** 2, axis=1) + s_2) *
                    (np.sum(arg_2 ** 2, axis=1) + s_2))

    def __repr__(self):
        """Return the python representation of the kernel."""
        obj_repr = "NormalizedLinearKernel("
        if self.offset != self.default_offset:
            obj_repr += f"offset={self.offset}"
        obj_repr += ")"
        return obj_repr


class HyperbolicKernel(Kernel):
    """Hyperbolic kernel class."""

    default_scale = 1
    default_offset = 0

    def __init__(self, scale=default_scale, offset=default_offset):
        r"""Create an instance of `HyperbolicKernel`.

        :param scale: scale constant, defaults to 1.
        :type scale: `float`
        :param offset: offset constant, defaults to 0.
        :type offset: `float`
        """
        super().__init__()
        self.scale = scale
        self.offset = offset

    def compute(self, arg_1, arg_2):
        r"""Compute the kernel value.

        The value $k(x_1, x_2)$ of a hyperbolic kernel is intended as the
        quantity $\tanh(\alpha x_1 \cdot x_2 + \beta)$, $\alpha$ and $\beta$
        being the scale and offset parameters, respectively. Given the two 
        arrays of kernels $Y$ and $Z$, the return value will be $(k(y_1,z_1), 
        k(y_2,z_2),..., k(y_m, z_m))$, $m$ being the number of elements in the
        arrays. 

        :param arg_1: First kernel array argument.
        :type arg_1: Object convertible to np.array
        :param arg_2: Second kernel array argument.
        :type arg_2: Object convertible to np.array
        :returns: `array` -- kernel values.
        """
        dot_orig = np.sum(np.array(arg_1) * np.array(arg_2), axis=1)
        return np.tanh(self.scale * dot_orig + self.offset)

    def __repr__(self):
        """Return the python representation of the kernel."""
        obj_repr = "HyperbolicKernel("
        if self.scale != self.default_scale:
            obj_repr += f"scale={self.scale}, "
        if self.offset != self.default_offset:
            obj_repr += f"offset={self.offset}, "

        if obj_repr.endswith(", "):
            return obj_repr[:-2] + ")"
        else:
            return "HyperbolicKernel()"


class PrecomputedKernel(Kernel):
    """Precomputed kernel class."""

    def __init__(self, kernel_computations):
        r"""Create an instance of `PrecomputedKernel`.

        :param kernel_computations: kernel computations.
        :type kernel_computations: square matrix of float elements
        :raises: ValueError if `kernel_computations` is not a square
          bidimensional array.
        """
        super().__init__()
        try:
            (rows, columns) = np.array(kernel_computations).shape
        except ValueError:
            raise ValueError('The supplied matrix is not array-like ')

        if rows != columns:
            raise ValueError('The supplied matrix is not square')

        self.kernel_computations = np.array(kernel_computations)
        

    def compute(self, arg_1, arg_2):
        r"""Compute the kernel value.

        The value of a precomputed kernel is retrieved according to the indices
        of the corresponding objects. Note that each index should be enclosed
        within an iterable in order to be compatible with sklearn.

        :param arg_1: First kernel array argument.
        :type arg_1: Object convertible to np.array
        :param arg_2: Second kernel array argument.
        :type arg_2: Object convertible to np.array
        :returns: `array` -- kernel values.
        """
        
        arg_1 = to_list(arg_1)
        arg_2 = to_list(arg_2)

        return [self.kernel_computations[a[0], b[0]]
                for a in arg_1 for b in arg_2]

        # arg_1 = arg_1.reshape(len(arg_1), 1)
        # z = (np.array(list(zip_longest(arg_1, arg_2,fillvalue=arg_1[0])))
        #        .reshape(len(arg_2), 2))

        # return self.kernel_computations[z[:,0], z[:,1]].reshape(len(arg_2),)
        # #return (np.array([self.kernel_computations[x,y] for x,y in z])
        # #          .reshape(len(arg_2),))

    def __repr__(self):
        """Return the python representation of the kernel."""
        return f"PrecomputedKernel({self.kernel_computations})"

    def __eq__(self, other):
        """Check kernel equality w.r.t. other objects."""
        return type(self) is type(other) \
            and np.array_equal(self.kernel_computations,
                               other.kernel_computations)
