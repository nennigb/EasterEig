# -*- coding: utf-8 -*-

# This file is part of eastereig, a library to locate exceptional points
# and to reconstruct eigenvalues loci.

# Eastereig is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Eastereig is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Eastereig.  If not, see <https://www.gnu.org/licenses/>.

"""
# Define CharPol class and auxilliary classes.

This class implements the **Partial Characteristic Polynomial**.

For eigenvalue problem which are analytic function of the parameter, the
characteristic polynomial is also an analytic function of the parameter.
However, the access to this polynomial is generally impossible because of its size.

An practical alternative is to introduce the concept of a Partial Characteristic Polynomial (PCP)
whose coefficients are regular functions in a large domain of the parameter space
and from which a subset of selected eigenvalues can be straightforwardly recovered at a negligible cost.

This polynomial is built from Vieta formulas and from the successive
derivatives of a selected set of eigenvalues.
[Vieta' formulas](https://en.wikipedia.org/wiki/Vieta%27s_formulas)

For more information about theoretical aspects see https://arxiv.org/abs/2505.06141.
Be careful, in the code, ordering of sequences may be different from the paper.
"""
import numpy as np
from numpy.linalg import norm
import numpy.polynomial as pol
import itertools as it
from collections import deque
from eastereig.eig import AbstractEig
from eastereig.utils import (_outer, diffprodTree, div_factorial, diffprodMV,
                             two_composition, Taylor)
from eastereig.loci import Loci
from eastereig.options import gopts
import sympy as sym
import scipy.linalg as spl
import scipy.optimize as spo
from scipy.special import factorial
import time
import tqdm
# // evaluation in newton method
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pickle
# multidimentional polyval
# /!\ numpy.polynomial.polynomial.polyval != np.polyval
# from numpy.polynomial.polynomial import polyval
# import numpy.polynomial.polyutils as pu
from eastereig.fpoly import polyvalnd
from matplotlib import pyplot as plt


def _dprod(order, S_dcoefs, C_dcoefs, max_order, shape, N):
    """Inner function used in Multiply."""
    dcoefs_order = np.zeros(shape, dtype=complex)
    # Get all combinaisons of with 2 coefficients to have exactly the given order
    for index in two_composition(order, max_order):
        # Compute the values derivative of the product
        dcoefs_order += diffprodTree([S_dcoefs[max_order[0] - index[0]],
                                      C_dcoefs[max_order[1] - index[1]]], N)
    return dcoefs_order


class _SequentialExecutor():
    """Context manager that mimic ProcessPoolExecutor without new process.

    Could work even if charpol involved petsc objects.
    """

    def __init__(self, *args,  **kargs):
        pass

    def __enter__(self, *args, **kargs):
        return self

    def __exit__(*exc_info):
        pass

    @staticmethod
    def map(fun, *iterable, **kargs):
        """Wrap the builtin function `map`."""
        return map(fun, *iterable)


class CharPol():
    r"""
    Provide a numerical representation of the Partial Characteristic Polynomial.

    This approach is suitable when more that two eigenvalue are close to coalesce or when
    several parameters are involved. In the following, the number of parameter is N.

    This class allows to :

      1. Reconstruct eigenvalue loci
      2. Locate EPs between the eigenvalues from the set.

    Partial Characteristic Polynomialis is built from Vieta formulas and from the successive
    derivatives of a selected set of L eigenvalues \(\Lambda\).

    This polynomial can be seen as a polynomial in \(\lambda\), whom coefficients \(a_n\\)
    are Taylor series in \(\nu_0, ..., \nu_m \).

    The PCP coefficients are thus expanded as follows

    $$
    \newcommand{\nuv}{\boldsymbol{\nu}}
    \newcommand{\alphav}{\boldsymbol{\alpha}}
    {a_k}(\nuv) \approx \mathcal{T}_{a_k}(\nuv) = \sum_{\alpha_1=0}^{D_1} \cdots
    \sum_{\alpha_N=0}^{D_N} (\hat{a}_k)_{\alphav} \, (\nuv_0-\nuv)^{\alphav} \quad
    \mathrm{where} \quad (\hat{a}_k)_{\alphav} = \frac{\partial^{\alphav}a_k(\nuv_0)}{\alphav!}.
    $$

    Here, we the multi-index notation $\alphav  = (\alpha_1, \dots, \alpha_N)$ and $D_i$ is the derivtive
    order of the parameter $\nu_i$.

    The partial characteristic polynomial reads
    $$p(\lambda, \boldsymbol \nu) = \sum_{k=0}^{L} a_k(\boldsymbol \nu) \lambda^{L-k}.$$
    By construction, $a_0(\boldsymbol \nu)=1$ is associated to $\lambda^L$. These coefficients
    are stored in descending order in the `dcoefs` attribut.
    """

    def __init__(self, dLambda, nu0=None, vieta=True):
        r"""Initialize with a list of eigenvalue derivatives or a list of `Eig` objects.

        Parameters
        ----------
        dLambda : list
            List of the derivatives of the eigenvalues used to build this object.
        nu0 : iterable
            The value where the derivatives are computed.
        vieta : bool, optional, default True
            Compute or not the CharPol coefficients using Vieta formula.
            Setting `vieta = False` can be useful for alternative constructors.

        Attributes
        ----------
        dLambda : list
            List of the derivatives of the eigenvalues used to build this object
        nu0 : iterable
            The value where the derivatives are computed
        dcoefs : iterable
            The taylor coefficients $a_n(\boldsymbol \nuv)$ ordered in descending order.
        N : int
            The number of eigs used in the charpol object.
        """
        # Initial check on input are delegate
        self.dLambda, self.nu0 = self._initial_init_checks(dLambda, nu0)
        # Compute the polynomial coef
        if vieta:
            self.dcoefs = self.vieta(self.dLambda)
            for i, c in enumerate(self.dcoefs):
                self.dcoefs[i] = np.asfortranarray(div_factorial(c))
            # Finalize the instanciation
            self._finalize_init()

    @staticmethod
    def _initial_init_checks(dLambda, nu0):
        """Perform initial checks on input parameters."""
        if isinstance(dLambda[0], np.ndarray):
            if nu0 is None:
                raise ValueError('If initialized by a list of array, `nu0` is mandatory.')
        elif isinstance(dLambda[0], AbstractEig):
            try:
                # Check for nu0. This attribute is created in `getDerivative*` method.
                nu0 = dLambda[0].nu0
            except AttributeError:
                print('Need to call `getDerivative*` before init a CharPol instance.')
            dLambda = [np.array(vp.dlda) for vp in dLambda]
        else:
            TypeError('Unsupported type for `dLambda` elements.')

        # Convert nu0 as an np.Array
        try:
            _ = iter(nu0)
            nu0_ = np.array(nu0, dtype=complex)
        except TypeError:
            nu0_ = np.array([nu0], dtype=complex)

        return dLambda, nu0_

    def _finalize_init(self):
        """Finalize CharPol initialization."""
        # Compute usefull coefficients for jacobian evaluation
        self._jacobian_utils()

        # How many eig in the charpol
        self.N = len(self.dcoefs) - 1

        # Store degree of Taylor expannsion in all variables
        self._an_taylor_order = tuple(np.array(self.dLambda[0].shape) - 1)

    @classmethod
    def _from_dcoefs(cls, dcoefs, dLambda, nu0):
        """Create CharPol from its polynomial coefficients.

        Parameters
        ----------
        dcoefs : list
            List of the derivatives of each coef of the polynom. It is assumed that
            a_N=1. The coefficaients are sorted in decending order such,
            1 lda**N + ... + a_0 lda**0.
        dLambda : list
            List of the derivatives of the eigenvalues used to build this
            object.
        nu0 : iterable
            The value where the derivatives are computed.

        Returns
        -------
        C : CharPol
            An new CharPol instance.
        """
        C = cls(dLambda, nu0, vieta=False)
        C.dcoefs = dcoefs
        C._finalize_init()
        return C

    @classmethod
    def from_recursive_mult(cls, dLambda, nu0=None, block_size=3):
        """Create recursively CharPol from the lambdas.

        The methods is based on _divide and conquer_ approach. It combines
        the Vieta's formulas and polynomial multiplications to speed up the
        computation. **This is the prefered approach**.

        Parameters
        ----------
        dLambda : list
            List of the derivatives of the eigenvalues used to build this object
        nu0 : iterable
            The value where the derivatives are computed
        block_size : int, optional, default 3
            The number of eigenvalue used in initial Vieta computation.

        Returns
        -------
        CharPol
            The new CharPol instance.
        """
        # Initial check on inputs and convert them to array
        dLambda, nu0 = cls._initial_init_checks(dLambda, nu0)
        # Create queue of dLambda block.
        num_block = len(dLambda) // block_size
        if num_block == 0:
            # splitting require at least 1 block
            num_block = 1
        dLambda_blocks = np.array_split(dLambda, num_block)
        block_list = deque(range(0, len(dLambda_blocks)))
        # Create a dictionnary containing CharPol for each dLambda_blocks
        # Vieta is used in each block
        prod = dict()
        for block, dLambda_in_block in zip(block_list, dLambda_blocks):
            prod[block] = cls(dLambda_in_block, nu0)

        # Consume the queue by computing the product of successive pairs
        while len(block_list) > 1:
            # Get 2 items from the left
            pair = (block_list.popleft(), block_list.popleft())
            # Add the pair to the right to consume it after
            block_list.append(pair)
            # Store the results in the dict
            prod[pair] = prod[pair[0]] * prod[pair[1]]

        # Return the final product
        return prod[block_list[0]]

    def __repr__(self):
        """Define the representation of the class."""
        return "Instance of {}  @nu0={} with #{} derivatives and #{} eigs.".format(self.__class__.__name__,
                                                                                   self.nu0,
                                                                                   self._an_taylor_order,
                                                                                   self.N)

    def export(self, filename):
        """Export a charpol object using pickle.

        Parameters
        ----------
        filename : string
            The full path to save the data.
        """
        with open(filename, 'bw') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a charpol object from pickle.

        Parameters
        ----------
        filename : string
            The full path to load data from.

        Returns
        -------
        C : CharPol
            The loaded CharPol object.
        """
        with open(filename, 'br') as f:
            C = pickle.load(f)
        return C

    def lda(self):
        """Return the eigenvalue value used in the CharPol."""
        Lda = np.array([lda.flat[0] for lda in self.dLambda])
        return Lda

    def coef0(self):
        """Return the coefficient of the CharPol @ nu0."""
        an = np.array([dcoef.flat[0] for dcoef in self.dcoefs])
        return an

    @staticmethod
    def vieta(dLambda):
        """Compute the sucessive derivatives of the polynomial coefficients knowing
        its roots and their successive derivatives.

        The methods is based on the Vieta formulas see for instance
        [Vieta' formulas](https://en.wikipedia.org/wiki/Vieta%27s_formulas)

        Parameters
        ----------
        dLambda : list
            List of the polynom roots successive derivatives. The input format is
            [D^(alpha) lda1, D^(alpha) lda2, ...], where alpha represent the
            multi-index with the derivation order for each variable.
            D^(alpha) lda1 is an ndarray and the first element [0, ...] is lda1.
            All D^(alpha) lda_i should have the same shape.

        Returns
        -------
        dcoef_pol : list
            list of the derivatives of each coef of the polynom. It is assumed that
            a_N=1. The coefficaients are sorted in decending order such,
            1 lda^N + ... + a_0 lda^0

        Examples
        --------
        Consider x^2 - 3*x + 2 = (x-1)*(x-2), where all roots are constant (0 derivatives)
        >>> dlambda=[np.array([[2, 0], [0, 0]]), np.array([[1, 0], [0, 0]])]
        >>> c = CharPol.vieta(dlambda)
        >>> norm(np.array([c[0].flat[0], c[1].flat[0], c[2].flat[0]]) - np.array([1, -3, 2])) < 1e-10
        True

        """
        # FIXME probably a problem if the number of lda is not even

        # Number of derivative is N - 1
        N = np.array(dLambda[0].shape) - 1
        # M number of eigenvalue in the set
        M = len(dLambda)
        # Intialize the coef of the polynomial
        dcoef_pol = []
        dcoef0 = np.zeros_like(dLambda[0], dtype=complex)
        dcoef0.flat[0] = 1.
        dcoef_pol.append(dcoef0)

        # Loop to create each coefficients of the polynomial depeding of the order
        for order in range(1, M+1):
            lda_id = np.arange(0, M)
            # local coeff for summation
            dcoef_pol_ = np.zeros_like(dLambda[0], dtype=complex)

            # TODO add test if no product ?
            # get all combinaison of with 1 eigenvalue , 2 etc....
            for index in it.combinations(lda_id, order):
                # Compute the values derivative of the product
                dcoef_pol_ += diffprodTree([dLambda[i] for i in index], N)

            dcoef_pol.append(dcoef_pol_*(-1)**order)

        # Return the successive derivative of the polynomial coef, no factorial !!
        return dcoef_pol

    def multiply(self, C, max_workers=None):
        """Multiply this polynomial by another Charpol Object.

        The two CharPol object must be computed at the same `nu0` value and the
        `dLambda` attributs should have the same format and the same number
        of derivatives.

        The derivatives of all coefficients are obtained from the coefficients
        of each polynomial.

        Depending of the number of terms, this appraoch may be faster than
        using Vieta formula when the number of eigenvalue increases.

        The number of process involved in the computation is defined in EasterEig
        options file and can be set using `ee.options.gopts['max_workers_mult'] = 4`.
        The default is 1. For better performance, use the number of cores.
        If `eig` object involve petsc matrices and collective executions,
        it may fail with value different from 1.

        Parameters
        ----------
        C : CharPol
            The second polynomial.
        max_workers : int, optional
            The number of process involved in the computation. The default is `None`.
            If `None` it use the value from `ee.options.gopts['max_workers_mult']`.
            This argument is mainly intent for testing.

        Returns
        -------
        P : CharPol
            The results of the multiplcation of both CharPol.
        """
        if not isinstance(C, self.__class__):
            raise ValueError('`C` must be a CharPol object.')
        # Need to check if both nu0 are the same
        try:
            if not np.allclose(self.nu0, C.nu0):
                raise ValueError('Both polynomials must be obtained at the same `nu0`')
        except AttributeError:
            print('The attribut `nu0` should be defined and identical for both polynomial')
        # Check derivatives order are the same
        if self.dcoefs[0].shape != C.dcoefs[0].shape:
            raise ValueError('The number of derivative should be the same for both polynomial')
        # Use the default the number of workers
        if max_workers is None:
            max_workers = gopts['max_workers_mult']
        # If max_workers is 1, define a context manager to avoid the initialization
        # of the ProcessPoolManager
        if max_workers == 1:
            PoolExecutor = _SequentialExecutor
        else:
            PoolExecutor = ProcessPoolExecutor

        total_order = len(self.dcoefs) + len(C.dcoefs) - 2
        max_order = (len(self.dcoefs) - 1, len(C.dcoefs) - 1)
        shape = self.dcoefs[0].shape
        # Deduce the highest degrees for derivatives
        N = np.array(shape) - 1

        # Get factorial
        F = np.asfortranarray(div_factorial(np.ones(shape, dtype=float)))
        F1 = 1. / np.asfortranarray(div_factorial(np.ones(shape, dtype=float)))

        dcoefs = []
        # dcoefs are Taylor series, need to remove the factorial for the derivatives
        C_dcoefs = [F1 * an for an in C.dcoefs]
        S_dcoefs = [F1 * an for an in self.dcoefs]
        # Compute the polynomial coef `dcoefs`
        # In CharPol these coefs are in descending order such,
        #    a_0 lda**N + ... + a_N lda**0
        __dprod = partial(_dprod, S_dcoefs=S_dcoefs, C_dcoefs=C_dcoefs,
                          max_order=max_order, shape=shape, N=N)
        order_range = np.arange(total_order, -1, -1)
        with PoolExecutor(max_workers=max_workers) as executor:
            # for results in map(__dprod, order_range):
            for results in executor.map(__dprod, order_range):
                # Local coeff for summation
                dcoefs_order = results
                # dcoefs should be a Taylor series, need to divide by factorial
                dcoefs.append(dcoefs_order * F)

        # Merge both dLambda for the new instance
        dLambda = []
        dLambda.extend(self.dLambda.copy())
        dLambda.extend(C.dLambda.copy())

        return CharPol._from_dcoefs(dcoefs, dLambda, self.nu0)

    def __mul__(self, other):
        """Multiply this polynomial by another Charpol Object.

        Shorthand or the `multiply` method.
        """
        return self.multiply(other)

    def conj(self):
        """Return a copy of the complex conjugate CharPol.

        Remarks
        -------
        This may be usefull to speed-up CharPol construction when all
        eigenvalues come in a complex conjugate pairs.
        """
        dcoefs = [ai.copy().conj() for ai in self.dcoefs]
        dLambda = [ai.copy().conj() for ai in self.dLambda]
        return CharPol._from_dcoefs(dcoefs, dLambda, self.nu0)

    def trunc(self, param_truncation=-1):
        """Return a copy of a truncated CharPol.

        Parameters
        ----------
        param_truncation: iterable or int
            This truncation parameter is negative and indicate how many derivative
            order should be removed.
            If it is a integer, the same truncation is applied for all
            parameters, else the truncation may be given for each variable.
            The default is `-1` and remove the last order. Use `None` in the
            iterable to pick all available order for a given variable.

        Remarks
        -------
        This is usefull to test sensitivity and convergence.
        """
        N = len(self.nu0)
        # Create slices accounting for truncation
        if isinstance(param_truncation, int):
            slices = (slice(0, param_truncation),) * (N )
        elif hasattr(param_truncation, '__iter__'):
            slices = tuple([slice(0, i) for i in param_truncation])
        else:
            raise ValueError('`param_truncation` should be an integer or an iterable')

        dcoefs = [ai[slices].copy() for ai in self.dcoefs]
        dLambda = [ai[slices].copy() for ai in self.dLambda]
        return CharPol._from_dcoefs(dcoefs, dLambda, self.nu0)

    @staticmethod
    def _set_param(index, value, an):
        """Compute the a new polynomial by setting one of its parameters.

        After moving the axes, the parameters remains in the same order but
        without the set parameter.
        The evaluation is based on `numpy.polynomial.polyval` and use ascending
        power ordering.
        The remaining indices of an enumerate multiple sets of coefficients
        corresponding to the unset coefficient.

        Examples
        --------
        >>> rng = np.random.default_rng(seed=1)
        >>> a = rng.random((4, 3, 2))
        >>> b1 = CharPol._set_param(1, 0.1, a)
        >>> b_direct = np.polynomial.polynomial.polyval3d(0.2, 0.1, 0.15, a)
        >>> np.allclose(b_direct, np.polynomial.polynomial.polyval2d(0.2, 0.15, b1))
        True
        >>> b2 = CharPol._set_param(2, 0.15, a)
        >>> np.allclose(b_direct, np.polynomial.polynomial.polyval2d(0.2, 0.1, b2))
        True
        >>> b0 = CharPol._set_param(0, 0.2, a)
        >>> np.allclose(b_direct, np.polynomial.polynomial.polyval2d(0.1, 0.15, b0))
        True
        """
        cn = np.moveaxis(an, index, 0)
        b = np.polynomial.polynomial.polyval(value, cn)
        return b

    def set_param(self, index, value):
        """Return a new CharPol with one of its parameters set to `value`.

        The parameters remains in the same order but without the set parameter.

        Parameters
        ----------
        index: int
            The index of the parameter to set.
        value: complex
            The value to set.

        Returns
        -------
        : Charpol
            The new charpol with a parameter of.
        """
        dcoefs_ = []
        dLambda_ = []
        value -= self.nu0[index]
        for an in self.dcoefs:
            dcoefs_.append(self._set_param(index, value, an))
        # Factorial are not included in dLambda, just apply it in the set direction
        shape = np.ones(len(self.nu0), dtype=int)
        shape[index] = self.dLambda[0].shape[index]
        fact = factorial(np.arange(0, shape[index]))
        fact = fact.reshape(shape)
        for ldan in self.dLambda:
            ldan_ = ldan / fact
            dLambda_.append(self._set_param(index, value, ldan_))

        nu0_ = np.asarray(self.nu0)[np.arange(self.nu0.size) != index]
        return CharPol._from_dcoefs(dcoefs_, dLambda_, nu0_)

    def EP_system(self, vals, trunc=None):
        """Evaluate the successive derivatives of the CharPol with respect to lda.

        This system is built on the sucessive derivatives of the partial
        characteristic polynomial,

        v = [p0, p1, ... pn]^t, where pi = d^ip0/dlda^i.

        Generically, this système yields to a finite set of EP.

        Parameters
        ----------
        vals : iterable
            Containts the value of (lda, nu_0, ..., nu_n) where the polynom
            must be evaluated. Althought nu is relative to nu0, absolute value
            have to be used.
        trunc : int or None
            The truncation to apply in each parameter wtr the maximum available
            order of the Taylor expansion. The integer is supposed
            to be negative as in `an[:trunc, :trunc, ..., :trunc]`.
            This argument is useful to compare several order of Taylor
            expansion.

        Returns
        -------
        v : np.array
            Value of the vectorial function @ vals
        """
        # Total number of variables (N-1) nu_i + 1 lda
        N = len(vals)
        # Create slices accounting for truncation
        slices = (slice(0, trunc),) * (N - 1)
        # polynomial degree in lda
        deg = len(self.dcoefs)
        # Extract input value
        lda, *nu = vals
        nu = np.array(nu, dtype=complex) - np.array(self.nu0, dtype=complex)
        # compute the an coefficients at nu
        an = np.zeros((len(self.dcoefs),), dtype=complex)
        # Compute a_n at nu
        for n, a in enumerate(self.dcoefs):
            # an[n] = pu._valnd(polyval, a, *nu)
            an[n] = polyvalnd(nu, a[slices])

        # Compute the derivtaive with respect to lda

        # Create a coefficient matrix to account for lda derivatives
        # [[1, 1, 1, 1], [3, 2, 1, 1], [2, 1, 1, 1]
        DA = np.ones((N, deg), dtype=complex)
        for i in range(1, N):
            DA[i, :-i] = np.arange((deg-i), 0, -1)
        # Evaluate each polynom pi
        v = np.zeros((N,), dtype=complex)
        for n in range(0, N):
            # apply derivative with respect to lda
            dan = an[slice(0, deg-n)] * np.prod(DA[0:(n+1), slice(0, deg-n)], 0)
            # np.polyval start by high degree
            # np.polynomial.polynomial.polyval start by low degree!!!
            v[n] = polyvalnd(lda, dan[::-1])

        return v

    def _jacobian_utils(self):
        """Precompute the shifted indices and combinatoric elements used
        in jacobian matrix (N+1) x (N+1).

        Attributes
        ----------
        _dlda_prod : np.array
            Coefficients used in 1D (lda) polynomial derivative. Assume decending order.
        _der_coef : np.array
            Coefficients used in ND (nu_0, ..., nu_N) polynomial derivative.
        _da_slice : np.array
            List of slices of da array.
        _da_shape : np.array
            List of shape of da array.
        """
        # get number of Parameters
        N = len(self.dcoefs[0].shape) + 1
        # polynomial degree in lda
        deg = len(self.dcoefs)
        # shape of a(nu)
        shape = np.array(self.dcoefs[0].shape)
        # Create a coefficient matrix dlda_prod to account for lda derivatives
        # [[1, 1, 1, 1], [3, 2, 1, 1], [2, 1, 1, 1]
        DA = np.ones((N+1, deg), dtype=complex)
        for i in range(1, N+1):
            DA[i, :-i] = np.arange((deg-i), 0, -1)
        self._dlda_prod = np.cumprod(DA, 0)

        # Create Coefficients used nu_i derivatives
        self._der_coef = np.empty((N, N), dtype=object)
        self._da_slice = np.empty((N, N), dtype=object)
        self._da_shape = np.empty((N, N), dtype=object)
        # Loop of the derivative of P, fill J raws
        for row in range(0, N):
            # loop over the nu variables, fill column
            for col in range(0, N):
                # store coef for lda-evaluation
                # recall that an[0] * lda**n
                an = np.zeros((len(self.dcoefs),), dtype=complex)
                # create the matrix accounting for derivative of coefs
                der_coef_list = []
                start = np.zeros_like(shape)
                # shape of the derivatives of the coef
                da_shape = shape.copy()
                if col > 0:
                    da_shape[col-1] -= 1
                    start[col-1] += 1
                da_slice_list = []
                for nu_i in range(0, N-1):
                    if nu_i == col - 1:
                        der_coef_list.append(np.arange(1, da_shape[nu_i]+1))
                        da_slice_list.append(slice(1, da_shape[nu_i]+1))
                    else:
                        der_coef_list.append(np.ones((da_shape[nu_i],)))
                        da_slice_list.append(slice(0, da_shape[nu_i]))
                # Maintains fortran ordering
                self._der_coef[row, col] = np.asfortranarray(_outer(*der_coef_list))
                self._da_slice[row, col] = tuple(da_slice_list)
                self._da_shape[row, col] = da_shape.copy()

    def jacobian(self, vals, trunc=None):
        """Compute the jacobian matrix of the EP system at nu.

        This system is built on the sucessive derivatives of the partial
        characteristic polynomial,
        ```
        J = [[dp0/dlda, dp0/dnu_0 ...., dp0/dnu_n],
             [dp1/dlda, dp1/dnu_0 ...., dp1/dnu_n],
              .....]
        ```
        where `i = d^ip0/dlda^i.`

        Generically, this system yields to a finite set of EP.

        Parameters
        ----------
        vals : iterable
            The value of the extended parameter (lda, nu_0, ..., nu_n) for which the polynom
            must be evaluated. Althought nu is relative to nu0, absolute value
            have to be used.
        trunc : int or None
            The truncation to apply in each parameter wtr the maximum available
            order of the Taylor expansion. The integer is supposed
            to be negative as in `an[:trunc, :trunc, ..., :trunc]`.
            This argument is useful to compare several order of Taylor
            expansion.

        Returns
        -------
        J : np.array
            The jacobian matrix.

        Notes
        -----
        Works with _jacobian_utils method. This methods will create some required
        indices and constants array.

        """
        # Total number of variables (N-1) nu_i + 1 lda
        N = len(vals)
        # Create slices accounting for truncation
        slices = (slice(0, trunc),) * (N - 1)
        # polynomial degree in lda
        deg = len(self.dcoefs)
        # Extract input value
        lda, *nu = vals
        nu = np.array(nu, dtype=complex) - np.array(self.nu0, dtype=complex)
        J = np.zeros((N, N), dtype=complex)

        # Alias coefficient matrix dlda_prod to account for lda derivatives
        dlda_prod = self._dlda_prod
        # Alias coefficients and indices matrices to account for nu derivatives
        der_coef = self._der_coef
        da_slice = self._da_slice
        da_shape = self._da_shape

        # Loop of the derivative of P, fill J raws
        for row in range(0, N):
            # loop over the nu variables, fill column
            for col in range(0, N):
                # store coef for lda-evaluation
                # recall that an[0] * lda**n
                an = np.zeros((len(self.dcoefs),), dtype=complex)
                # Loop over the polynomial coefs
                for n, a in enumerate(self.dcoefs):
                    # Recall that a[0,...,0] * nu0**0 * ... * nu_m**0
                    # Create a zeros matrix
                    # Maintains fortran ordering
                    da = np.zeros(da_shape[row, col], dtype=complex, order='F')
                    # and fill it with the shifted the coef matrix
                    da = a[da_slice[row, col]] * der_coef[row, col]
                    # an[n] = pu._valnd(polyval, da, *nu)
                    an[n] = polyvalnd(nu, da[slices])
                # apply derivative with respect to lda
                if col == 0:
                    # Increase the derivation order
                    # dan = an[0:-(row+1)] * np.prod(DA[1:(row+2), :-(row+1)], 0)
                    dan = an[0:-(row+1)] * dlda_prod[row+1, :-(row+1)]
                else:
                    # Apply successived derivative of the parial Char pol
                    # dan = an[slice(0, deg-row)] * np.prod(DA[0:(row+1), slice(0, deg-row)], 0)
                    dan = an[slice(0, deg-row)] * dlda_prod[row, slice(0, deg-row)]
                    # np.polyval start by high degree
                    # np.polynomial.polynomial.polyval start by low degree!!!
                # J[row, col] = polyval(lda, dan[::-1])
                J[row, col] = polyvalnd(lda, dan[::-1])
        return J

    def eval_at(self, vals):
        """Evaluate the partial caracteristic polynomial at (lda, nu).

        Parameters
        ----------
        vals : iterable
            Containts the value of (lda, nu_0, ..., nu_n) where the polynom
            must be evaluated. Althought nu is relative to nu0, absolute value
            have to be used.

        Returns
        -------
        The partial caracteristic polynomial at (lda, nu).

        """
        # Extract input value
        lda, *nu = vals
        # Evaluate the polynomial *coefficient* at nu
        an = self.eval_an_at(nu)
        # Evaluate the polynomial
        # np.polyval start by high degree
        # np.polynomial.polynomial.polyval start by low degree!!!
        return polyvalnd(lda, an[::-1])

    def eval_an_at(self, nu):
        """Evaluate the partial caracteristic polynomial coefficient an at nu.

        Parameters
        ----------
        nu : iterable
            Containts the value of (nu_0, ..., nu_n) where the polynom
            must be evaluated. Althought nu is relative to nu0, absolute value
            have to be used.

        Returns
        -------
        an : array
            The partial caracteristic polynomial coefficient at nu in descending order
            an[0] * lda **(n-1) + ... + an[n-1]
        """
        # Extract input value
        nu = np.array(nu, dtype=complex) - np.array(self.nu0, dtype=complex)
        an = np.zeros((len(self.dcoefs),), dtype=complex)
        # Compute a_n at nu
        for n, a in enumerate(self.dcoefs):
            # an[n] = pu._valnd(polyval, a, *nu)
            an[n] = polyvalnd(nu, a)

        # np.polyval start by high degree
        # np.polynomial.polynomial.polyval start by low degree!!!
        return an

    def eval_lda_at(self, nu):
        """Evaluate the eigenvalues at nu.

        Parameters
        ----------
        nu : iterable
            Containts the values of (nu_0, ..., nu_n) where the polynom
            must be evaluated. Althought nu is relative to nu0, absolute value
            have to be used.

        Returns
        -------
        lda : array
            The eigenvalue estimated at nu.
        """
        an = self.eval_an_at(nu)
        return np.roots(an)

    @staticmethod
    def _newton(f, J, x0, tol=1e-8, normalized=True, verbose=False):
        """Run basic Newton method for vectorial function.

        Parameters
        ----------
        f : function
            The vectorial function we want to find the roots. Must Returns
            an array of size N.
        J : function
            Function that provides the Jacobian matrix NxN.
        x0 : array
            The initial guess of size N.
        tol : float
            The tolerance to stop iteration.
        normalized : bool
            If `True` the tol is applied to the ratio of the ||f(x)||/||f(x0)||
        verbose : bool
            print iteration log.

        Returns
        -------
        x : array
            The solution or None of NiterMax has been reached or if the
            computation fail.
        """
        # set max iteration number
        NiterMax = 150
        x = np.array(x0)
        k = 0
        cond = True
        while cond:
            fx = f(x)
            if k == 0:
                norm_fx0 = np.linalg.norm(fx)
            # working on polynomial may leads to trouble if x is outside the
            # convergence radius. Use try/except
            try:
                x = x - np.linalg.inv(J(x)) @ fx
                if verbose:
                    print(k, x, abs(fx))
                k += 1
                # condition
                if normalized:
                    cond = (k < NiterMax) and (np.linalg.norm(fx)/norm_fx0) > tol
                else:
                    cond = (k < NiterMax) and np.linalg.norm(fx) > tol
            except:
                print('Cannot compute Newton iteration at x=', x)
                print('  when starting from x0=', x0)
                return None

        # if stops
        if k < NiterMax:
            return x
        else:
            print('The maximal number of iteration has been reached.')
            return None

    def lm_from_sol(self, z0, tol=1e-8, normalized=True, verbose=False):
        """Find EP with Levenberg-Marquard solver from single starting point.

        Based on scipy/MINPACK implementation of 'lm' through the scipy.optimize
        root function.
        Since the algorithm works with real value problem. Real and imaginary are
        split.

        Parameters
        ----------
        x0 : array
            The initial guess of size N.
        tol : float
            The tolerance to stop iteration.
        normalized : bool
            If `True` the tol is applied to the ratio of the ||f(x)||/||f(x0)||. Added
            for compatibility but not used.
        verbose : bool
            Print iteration log.

        Returns
        -------
        x : array
            The solution or None of NiterMax has been reached or if the
            computation fail.
        """
        # set max iteration number
        niter_max = 100
        # Def few local function
        def to_z(x):
            """Convert the in-lined real and imag unknown `x` to complex unknown."""
            return x.reshape((-1, 2)) @ np.array([1, 1j])

        def to_x(z):
            """Convert the complex unknown `z` toin-line real and imag unknown."""
            return np.stack((z.real, z.imag), axis=1).ravel()

        def f_and_jac(x):
            """Split the EP system and the jacobian matrix into real and imag."""
            # Compute the Jac and the function from charpol (C)
            nparam = len(z0)
            z = to_z(x)
            Jc = self.jacobian(z)
            Fc = self.EP_system(z)
            # Compute the real version
            Jr = np.zeros((x.size, x.size))
            Fr = np.zeros((x.size,))
            # Add terms from p dans d_lda p
            for eq_id in range(nparam):
                eq_idr = eq_id * 2
                Fr[eq_idr] = Fc[eq_id].real
                Fr[eq_idr + 1] = Fc[eq_id].imag
                for i in range(nparam):
                    ir = i * 2
                    # Real part eqs
                    Jr[eq_idr, ir] = Jc[eq_id, i].real
                    Jr[eq_idr, ir+1] = - Jc[eq_id, i].imag
                    # Imag part eqs
                    Jr[eq_idr+1, ir] = Jc[eq_id, i].imag
                    Jr[eq_idr+1, ir+1] = Jc[eq_id, i].real
            return Fr, Jr

        # see also hybr, sometimes beter ?
        # scaling = np.ones_like(x)
        # scaling[6:] = imag_scaling
        # Check input
        if isinstance(z0, np.ndarray):
            z0 = z0.ravel()
        else:
            z0 = np.array(z0).ravel()
        x = to_x(z0)
        # scaling = 1. # / np.abs(x)
        sol = spo.root(f_and_jac, x, method='lm', jac=True, callback=None,
                       options={'maxiter': niter_max, 'ftol': tol, 'xtol': tol})  # 'diag': scaling
        # })  #{}'xtol': 0.00000001,
        if verbose:
            print(sol)

        z = to_z(sol.x)
        x = sol.x
        if sol.success:
            if verbose:
                print('Convergence in {} iterations.'.format(sol.nfev), sol.message)
        else:
            print(sol.message)
            z = None
        return z

    def newton_from_sol(self, x, **kargs):
        """Find EP with Newton-Raphson solver from single starting point.

        See `iterative_solve` method for `kargs` description.

        Parameters
        ----------
        x : array
            The initial guess of size N.
        """
        return self._newton(self.EP_system, self.jacobian, x, **kargs)

    def newton(self, *args, **kargs):
        """Mesh parametric space and run iterative solver to find EP in parallel."""
        print('Method `Newton` is now deprecated, use now `iterative_solve`.')
        return self.iterative_solve(*args, **kargs)

    def iterative_solve(self, bounds, Npts=4, decimals=4, max_workers=4, tol=1e-4,
                        verbose=False, algorithm='nr', chunksize=100):
        """Mesh parametric space and run iterative solver to find EP in parallel.

        Parameters
        ----------
        bounds : iterable
            Each item must contains the 2 corners in the complex plane. For
            instance if bounds = [(-1-1j, 1+1j), (-2-2j, 2+2j)],
            the points will be put in this C**2 domain.
            Althought nu is relative to nu0, absolute value have to be used.
        Npts : int
            The number of point in each direction between the bounds.
        decimals : int
            The number of decimals keep to filter the solution (using np.unique).
            Note it should be concistant with `tol` (~|log10(tol)|).
        max_workers : int
            The number of workers to explore the parametric space.
        tol : float
            The tolerance to stop iteration. See solver details for deeper insight.
        normalized : bool
            If `True` the tol is applied to the ratio of the ||f(x)||/||f(x0)|| in
            `nr` solver.
        verbose : bool
            print iteration log.
        algorithm : string
            Can be either 'nr' for Newton-Raphson, either 'lm' for Levenberg-Marquard.
            The second is often more robust.
        chunksize : int
            The size of computation grouped together and sent to workers for
            parallel execution.

        Returns
        -------
        sol : array
            The N 'unique' solutions for the M unknowns in a NxM array.

        Remarks
        -------
        For parallel execution, consider using (before numpy import)
        ```python
        import os
        os.environ["OMP_NUM_THREADS"] = "1" # or MKL_NUM_THREADS on windows
        ```
        """
        if len(bounds) != len(self.nu0) + 1:
            raise IndexError('The length of `bounds` must match the number'
                             ' of unknown ({}).'.format(len(self.nu0) + 1))

        def param_mesh(bound, Npts):
            Re, Im = np.meshgrid(np.linspace(bound[0].real, bound[1].real, Npts),
                                 np.linspace(bound[0].imag, bound[1].imag, Npts))
            z = Re + 1j * Im
            return z.ravel()

        # TODO May limit bounds by convergence radius ?
        tic = time.time()
        # Create a coarse mesh of the parametric space for Newton solving
        grid = []
        # Strategy linsapce
        for i, bound in enumerate(bounds):
            if bound is None:
                if i == 0:
                    grid.append(self.lda())
                else:
                    raise ValueError('Only the eigenvalue bound can be `None`.')
            else:
                grid.append(param_mesh(bound, Npts))

        # For each, run Newton search
        all_sol = []

        if algorithm == 'nr':
            # use partial to fixed all function parameters except lda
            _p_solver = partial(self._newton, self.EP_system, self.jacobian, tol=tol,
                                verbose=verbose, normalized=normalized)
        elif algorithm == 'lm':
            _p_solver = partial(self.lm_from_sol, tol=tol, verbose=verbose)
        else:
            raise NotImplementedError('The request algorithm `{}` is not recognized.'.format(algorithm))
        total = np.prod([len(g) for g in grid])
        pbar = tqdm.tqdm(total=total, position=0, leave=True)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for s in executor.map(_p_solver,
                                  it.product(*grid),
                                  chunksize=chunksize):
                all_sol.append(s)
                pbar.update()

        # Filter all solution to keep only unique
        # Remove None and convert to array
        sol_ = np.array([s for s in all_sol if s is not None])
        # Use unique to remove duplicated row
        _, indu = np.unique(sol_.round(decimals=decimals), axis=0, return_index=True)
        print('> ', time.time() - tic, ' s in iterative `{}` solver.'.format(algorithm))
        sol = sol_[indu, :]
        return sol

    def homotopy_solve(self, degree=None, tracktol=1e-12, finaltol=1e-8, singtol=1e-14,
                       dense=True, bplp_max=2000, oo_tol=1e-5, only_bezout=False, tol_filter=-1):
        """Solve EP system by homotopy method.


        This method defines a simplified interface to `pypolsys` homotopy
        solver.
        Such solver allows to find **all solutions** of the polynomial
        systems.

        An upper bound of the number of solution is given by
        the Bezout number, equal to the product of the degrees each polynomials.
        This number may be huge. The number of paths is given
        by `bplp` and can be quickly obtained without performing the tracking by setting
        `only_bezout=True`.

        Here, we use a m-homogeneous variable partition to limit the number of
        spurious solutions 'at infinty' to track.

        This method is interesting because all solutions are found, whereas
        for iterative methods like Newton-Raphson or Levenberg-Marquard.

        Parameters
        ----------
        degree : iterable, optional
            The maximum degree to keep for each variable. Default is `None` and all
            terms are kept.
        tracktol : float, optional
            Is the local error tolerance allowed the path tracker along
            the path.
        finaltol : float, optional
            Is the accuracy desired for the final solution.  It is used
            for both the absolute and relative errors in a mixed error criterion.
        singtol : float, optional
            Is the singularity test threshold used by SINGSYS_PLP.  If
            SINGTOL <= 0.0 on input, then SINGTOL is reset to a default value.
        dense : bool, optional
            If `True`, evaluate the polynomial using the fastermultivariate Horner
            scheme, optimized for dense polynomial. Else, monomial evaluation
            are used.
        bplp_max : int, optional
            Provides an upper bound to run homotopy solving.
            If `bplp > bplp_max`, the computation is aborted.
        oo_tol : float, optional
            Tolerance to drop out solution at infinity. If `oo_tol=0`, all
            solution are returned.
        only_bezout : bool, optional
            If `True` just compute the Bezout number (fast) and exit.
        tol_filter: flaot, optional
            Tolerance to drop small terms in the charpol. To keep all terms,
            -1 can be used. The amplitude of leading order in lda**N is 1 by
            convention.

        Returns
        -------
        bplp : int
            The Bezout number corresponding to the number of tracked paths.
        r : np.array
            The solutions of the system express as absolute value wtr to nu0.
            `r` is `None` if the computation has been aborted. If `oo_tol>0`,
            the number of rows may be less than `bplp`.
            If `bplp > bplp_max`, `r` is None.

        Remarks
        -------
        Requires `pypolsys` python package.
        """
        try:
            import pypolsys
        except ModuleNotFoundError:
            print('The module `pypolsys` is not installed.',
                  ' Install it or use `iterative_solve` method.')
        # Convert to sympy
        p0, variables = self.taylor2sympyPol(self.dcoefs, tol=tol_filter)
        _lda, *_ = variables
        # Build the system
        polys = []
        _lda = p0.gens[0]
        n_var = len(p0.gens)
        # Truncate the serie if needed
        if degree is not None:
            p0 = self._drop_higher_degree(p0, degree)
        polys.append(p0)
        for i in range(1, len(variables)):
            polys.append(polys[i-1].diff(_lda))

        # Pypolsys solve
        # generate sparse polynomial
        t0 = time.time()
        pol = pypolsys.utils.fromSympy(polys)
        if dense:
            pol = pypolsys.utils.toDense(*pol)
        part = pypolsys.utils.make_mh_part(n_var, [[i] for i in range(1, n_var+1)])
        pypolsys.polsys.init_poly(*pol)
        pypolsys.polsys.init_partition(*part)

        # Check if there is too much solution
        bplp = pypolsys.polsys.bezout(singtol)
        print('> Bezout number :', bplp)
        if bplp > bplp_max:
            print('  `bplp` > `bplp_max` ({}). Abort. Increase `bplp_max` and try again.'.format(bplp_max))
            return bplp, None
        elif only_bezout:
            return bplp, None
        else:
            bplp = pypolsys.polsys.solve(tracktol, finaltol, singtol)
            r = pypolsys.polsys.myroots.copy()
            # Express the solution absolutly wrt nu0
            for i, nu in enumerate(self.nu0):
                r[i+1, :] += nu
            # keep only finite solution, filter point at oo
            finite_sol = np.nonzero(np.abs(r[-1, :]) > oo_tol)
            print('> ', time.time() - t0, 's in homotopy solve. Found', bplp, 'solutions.')
            return bplp, r[:-1, finite_sol[0]].T

    def filter_spurious_solution(self, sol, trunc=-1, filter=True, tol=1e-2,
                                 plot=False, sort=True):
        r"""Remove spurious solutions based on roots sensitivty estimation.

        The multivariate polynomial system leading to EPs can be recast into
        $$ \mathbf{S}(\mathbf{s}^*) = \mathbf{0}.$$

        The sensitivity indicator which corresponds to a single NR correction step starting from
        the solution $\mathbf{s}$ obtained using the maximum order of derivation $D$ in the
        Taylor expansion
        $$
            \delta = \Vert\mathbf{J}^{-1} \mathbf{S}(\mathbf{s})\Vert_2,
        $$
        where the Jacobian matrix $\mathbf{J}$ of the multivariate system are both computed
        using $D-1$ derivatives in the expansion.

        Parameters
        ----------
        sol : array
            Contains all the found EP candidates solutions along its rows.
        trunc : int, optional
            The truncation to apply in each parameter wtr the maximum available
            order of the Taylor expansion. The integer is supposed
            to be negative as in `an[:trunc, :trunc, ..., :trunc]`.
        filter : bool, optional
            If `True`, only true solution are filter.
        tol : float, optional
            The tolerance used in the filtering. sol such kappa(sol) > tol are
            removed.
        plot: bool, optional
            Plot the sensitivity distribution.
        sort: bool, optional
            If True, sort the filtered solution with ascending sensitivity
            `deltaf`. the default is `True`.

        Returns
        -------
        delta : array
            The estimated sensitivity of each `sol`.
        solf : array
            The filtered solution if `filter=True`, `None` otherwise.
        deltaf : array
            The sensitivity of the filtered solution if `filter=True`, `None` otherwise.
        """
        delta = np.zeros((sol.shape[0],))
        for i, soli in enumerate(sol):
            J = self.jacobian(soli, trunc=trunc)
            delta[i] = np.linalg.norm(np.linalg.inv(J) @ self.EP_system(soli, trunc=trunc))

        if filter:
            solf = sol[delta < tol, :].copy()
            deltaf = delta[delta < tol].copy()
            if sort:
                index = np.argsort(deltaf)
                deltaf = deltaf[index]
                solf = solf[index, :]
        else:
            solf = None
            deltaf = None

        if plot:
            fig_delta, ax_delta = plt.subplots()
            ax_delta.semilogy(sorted(delta), 'k+', markersize='6')
            ax_delta.axhline(y=tol, color='grey', linestyle=':')
            ax_delta.set_ylabel(r'$\delta$')
            ax_delta.set_xlabel('Solution index')
        return delta, solf, deltaf

    @staticmethod
    def _drop_higher_degree(p, degrees):
        """Remove higher degrees for the `sympy` Poly object.

        This method is useful when the polynomial stands for a Taylor series from
        which you want drop higher order terms.

        Parameters
        ----------
        p : Sympy Poly
            The polynom to truncate.
        degrees : iterable
            The maximum degree to keep for each variable.

        Returns
        -------
        Sympy Poly
            The truncated polynomial.
        """
        p_dict = p.as_dict()
        degrees = np.asarray(degrees)
        for key, val in list(p_dict.items()):
            if (np.array(key) > degrees).any():
                del p_dict[key]
        return sym.Poly.from_dict(p_dict, p.gens)

    @staticmethod
    def taylor2sympyPol(coef_list, tol=1e-12):
        """Convert a polynom into a sympy polynomial object in (nu, lda).

        Let us consider a_n(nu) lda**n + .... + a_0(nu), where nu stands for
        all variables.

        Parameters
        ----------
        coef_list : List
            Contains ndarray with the Taylor series of an [an, a_n-a, ..., a0]
        tol : float, optional
            The tolerance used to drop out the smaller terms.

        Returns
        -------
        P : Pol
            Multivariate polynomial in sympy format on complex field.
        variables : iterable
            The list of the sympy symbolic variables.

        Examples
        --------
        >>> a2, a1, a0 = np.array([[1, 2], [3, 6]]), np.array([[10, 0], [0, 0]]), np.array([[0, 5], [7, 0]])
        >>> P, variables = CharPol.taylor2sympyPol([a2, a1, a0])  # lda**2 a2 + lda * a1 + a0
        >>> print(P)
        Poly(6.0*lambda**2*nu0*nu1 + 3.0*lambda**2*nu0 + 2.0*lambda**2*nu1 + 1.0*lambda**2 + 10.0*lambda + 7.0*nu0 + 5.0*nu1, lambda, nu0, nu1, domain='CC')
        """
        # TODO is there a best order ?
        # FIXME add a truncation flag to sparsify
        coef_dict = dict()
        # Store all coefs in a dict²
        for n, c in enumerate(reversed(coef_list)):
            for index, dc in np.ndenumerate(c):
                if abs(dc) > tol:
                    key = (n, *index)
                    coef_dict[key] = dc

        # get the number of Parameters nu
        nu_dim = len(coef_list[0].shape)
        var_string = 'lambda, nu0:' + str(nu_dim)
        # Use the dict to create a sympy Poly expression
        variables = sym.symbols(var_string)
        P = sym.Poly(coef_dict, variables, domain='CC')
        return P, variables

    def getdH(self):
        r"""Compute the Taylor series of the partial discriminant H with respect to nu.

        H is proportional to the discriminant of the partial characteristic polynomial
        and vanishes for multiple roots like EPs.

        This method provides Taylor series of Discriminant, whereas the `discriminant`
        method, using the Sylvester matrix, just provides an evaluation at fixed
        nu values.
        This Taylor series may be particularly usefull when dealing with univariate
        polynomial to find all the roots.

        This _discriminant_ estimation is obtained from the eigenvalues used in the PCP,
        $$
        \newcommand{\nuv}{\boldsymbol{\nu}}
        \mathcal{H}(\nuv) = \prod_{1\le i<j\le L} (\lambda_i(\nuv)-\lambda_j(\nuv))^2
        $$
        for all \(\lambda_i \in \Lambda\),
        this expression can be obtained, seeing that
        $$
        H(\nu) = \prod_{p \in P} h_{p}(\nu),
        $$
        where \(h_{p}(\nu)=(\lambda_i(\nu)-\lambda_j(\nu))^2\) for \(i\neq j \).
        Here, we called \(P\) the set of all possible pair of eigenvalues from the set of the
        selected eigenvalue \(\Lambda\).

        Returns
        -------
        TH : Discr
            Object that contains the Taylor series of H.

        Remarks
        -------
        H is equal to the discriminant iff `an=1`, else you need to multiply by `an**(2*n -2)`.
        """
        # It will allow to elimitate on variable and continue to use homotopy
        dLambda = self.dLambda

        # assume all dLambda have the same shape
        N = np.array(dLambda[0].shape) - 1     # Number of derivative
        M = len(dLambda)                       # M number of eigenvalue in the set
        # Card of permution of 2 in M element
        # CM = int(sp.special.binom(M, 2))
        lda_id = np.arange(0, M)

        Index = []
        # size CM, contains derivative h(i pair)
        dh = []
        # Get all combinaison of eigenvalue and compute dh(i pair)
        for i in it.combinations(lda_id, 2):
            # strore index for later use
            Index.append(i)
            # Compute the values of dh for a pair :
            # f = (lda0 -  lda1)**2 = (lda0 -  lda1) * (lda0 -  lda1)
            f = dLambda[i[0]].squeeze() - dLambda[i[1]].squeeze()
            dh_ = diffprodMV([f, f], N)
            # use dedicated formulas for 1 variable, (less sumation)
            # dh_ = EP.dlda2dh(dLambda[i[0]].squeeze(),
            #                  dLambda[i[1]].squeeze())
            dh.append(dh_)

        # Derivate the product of h_i
        dH = diffprodTree(dh, N)
        # dH is the successive derivative, div by factorial to get TH
        TH = Discr(div_factorial(dH), self.nu0)
        return TH

    def associate_lda_to_nu(self, nu_ep, tol=1e-3):
        """Associate a `nu` obtained with `getdH` to its lda.

        Solve the Chapol using nu and check which lda yields to multiple roots
        using few NR steps. Direct appraoch based on `EP_system` fails due to
        round of error.

        Parameters
        ----------
        nu_ep : iterable or scalar
            Vector or parameters identified as an EP.
        tol : float, optional
            The tolerance used in NR refinement.

        Returns
        -------
        lda_ep : complex
            The EP eigenvalue. If the methods fails, it returns `None`.
        nu_ep : complex
            The NR refined EP value. If the methods fails, it returns `None`.
        lda_ep_id : int
            the index of the eigenvalue in the `Charpol` list.  If the methods
            fails, it returns `None`.

        # TODO add tests!
        """
        # Transform into tuple if scalar
        if not hasattr(nu_ep, '__iter__'):
            nu_ep = np.array((nu_ep,))
        # Find all lda at ep
        all_lda = np.roots(self.eval_an_at(nu_ep))
        # All lda solve Charpol, but at EP d_lda Charpol also vanish
        for i, ldai in enumerate(all_lda):
            nr_sol = self.newton_from_sol((ldai, *nu_ep), tol=tol, normalized=True)
            # Check if the refine sol correspond to the initial EP up to tol
            if (np.abs(nr_sol[1:] - nu_ep)/np.abs(nu_ep)).max() < 10*tol:
                lda_ep = ldai
                return lda_ep, nr_sol[1:], i
        return None, None, None

    def discriminant(self, nu):
        r"""Compute the discriminant from the coefficient and the Sylvester matrix.

        If we consider a polynom p, with coefficients an, the discriminant is given
        $$
        \operatorname Disc_{x}(p)={\frac {(-1)^{n(n-1)/2}}{a_{n}}}\operatorname {Res}_{x}(p,p')
                   = \frac {(-1)^{n(n-1)/2}}{a_{n}} \det S
        $$ where S is the `sylvester` matrix.

        Parameters
        ----------
        nu : iterable
            Contains the value (nu_0, nu_1, ...) where the discriminant is computed.

        Returns
        -------
        d : complex
        """
        # polynomial coef
        an = self.eval_an_at(nu)
        n = an.size - 1
        dan = an[:-1] * self._dlda_prod[1, :-1]
        # Sylvester need ascending order
        S = self.sylvester(an[::-1], dan[::-1])
        # S = self.sylvester(an, dan)
        d = spl.det(S) * (-1)**(n * (n-1)/2) * an[0]
        print(np.linalg.cond(S))
        Q, R = spl.qr(S)
        print('QR :', np.prod(np.diag(R)))
        return d

    def disc_EP_system(self, nu):
        r"""Evaluate the successive discriminant to locate higher order EP.

        The resultant, build as the determinant of the Sylvester matrix
        allows to check if two polynomials share roots. The following
        system can be used to check if an eigenvalue is common to the
        characteristic polynomial and its derivative with respect to \(\lambda\).
        In this case, it is equivalent to find the value \(\boldsymbol \nu\)
        where the discrimiant vanishes. This is the necessary condition to have
        an EP2.
        Applying this recursively, this yields to the following system,
        $$
        \begin{pmatrix}
        \mathrm{Res}_\lambda \big(p(\lambda; \boldsymbol \nu), \partial_\lambda p(\lambda; \boldsymbol \nu)\big) \\
        \mathrm{Res}_\lambda \big(\partial_{\lambda}p(\lambda; \boldsymbol \nu), \partial_{\lambda\lambda} p(\lambda; \boldsymbol \nu)\big) \\
        \vdots
        \end{pmatrix}
        $$
        By solving it, we can find \(\boldsymbol \nu\) leading to higher order EP.
        The number of line is equal to #nu.

        Parameters
        ----------
        vals : iterable
            Containts the value of (lda, nu_0, ..., nu_n) where the polynom
            must be evaluated. Althought nu is relative to nu0, absolute value
            have to be used.

        Returns
        -------
        v : np.array
            Value of the vectorial function @ nu

        Remarks
        -------
        This system is an alternative to `EP_system` based on the roots of the
        characteristic polynomial and its derivatives leading to \(\lambda, \boldsymbol \nu\).
        The discriminant allows to eliminate \(\lambda\) (or another variable).
        """
        # Compute the polynomial coef
        an = self.eval_an_at(nu)
        n = an.size - 1
        # Initialize output v
        v = np.zeros((len(self.dcoefs[0].shape),), dtype=complex)
        for i, vi in enumerate(v):
            if i == 0:
                p = an
            else:
                p = an[:-(i)] * self._dlda_prod[i, :-i]
            q = an[:-(1+i)] * self._dlda_prod[(1+i), :-(1+i)]
            # Sylvester need ascending order
            S = self.sylvester(p[::-1], q[::-1])
            # S = self.sylvester(an, dan)
            ni = n - i
            v[i] = np.linalg.det(S) * (-1)**(ni * (ni-1)/2) * p[0]
        return v

    @staticmethod
    def sylvester(p, q):
        r"""Compute the Sylvester matrix associated to p and q.

        https://en.wikipedia.org/wiki/Sylvester_matrix
        Formally, let p and q be two nonzero polynomials, respectively of degree m and n, such
        $$p(z) = p_0 + p_1 z + p_2 z^2 + \cdots + p_m z^m,$$
        $$q(z) = q_0 + q_1 z + q_2 z^2 + \cdots + q_n z^n.$$

        If the two polynomials have a (non constant) common factor. In such a case, the
        determinant of the associated Sylvester matrix, also called the resultant of
        the two polynomials vanishes. If q = dp / dz, the determinant of the Sylvester
        matrix is the discriminant.

        The Sylvester matrix associated to p and q is (n + m) × (n + m) is build
        on coefficients shift as shown in the example below.

        Parameters
        ----------
        p : array
            The coefficient of the polynomial p, sorted in ascending order.
        q : array
            The coefficient of the polynomial q, sorted in ascending order.

        Returns
        -------
        S : array
            Containing the Sylvester matrix (complex)

        Examples
        --------
        p = (z-2)**2 * (z-3), with a double roots 2
        >>> p = np.array([-12, 16, -7, 1])
        >>> q = np.array([16, -14, 3])
        >>> CharPol.sylvester(p, q).real
        array([[  1.,  -7.,  16., -12.,   0.],
               [  0.,   1.,  -7.,  16., -12.],
               [  3., -14.,  16.,   0.,   0.],
               [  0.,   3., -14.,  16.,   0.],
               [  0.,   0.,   3., -14.,  16.]])
        >>> np.abs(np.linalg.det(CharPol.sylvester(q, p))) < 1e-12
        True

        """
        m = p.size
        n = q.size
        # size of the Symvester matrix
        N = m + n - 2
        S = np.zeros((N, N), dtype=complex)
        # if q has higher degree, need to swap them
        if n > m:
            p, q = q, p
            m, n = n, m
        # fill upper part with p coefficients
        for i in range(0, n - 1):
            S[i, i:(i+m)] = p[::-1]
        # fill lower part with q coefficients
        for i in range(0, m - 1):
            S[i+n-1, i:(i+n)] = q[::-1]

        return S

    def radii_estimate(self, plot=False):
        """Estimate the radii of convergence of all CharPol coefficients.

        The approach is based on least square fit for all paremeters and
        all the CharPol coefficients and assume single variable dependance.

        Returns
        -------
        R : dict
            A dictionnary containing the statistics of the radius of convergence for
            all coefficients and along all directions.
            The primary keys are the parameter integer index and the condidary
            keys are the `mean`, `min` and the `std` obtained for all polynomial coefficients.
        """
        R = {}
        dcoef_r = np.zeros((self.dcoefs[0].ndim, len(self.dcoefs)-1))
        # Loop over chapol coef, skip 1st
        for n, an in enumerate(self.dcoefs[1:]):
            # Loop over the parameters
            for p in range(0, an.ndim):
                index = [0] * an.ndim
                index[p] = slice(0, None)
                dcoef_r[p, n] = Taylor._radii_fit_1D(an[tuple(index)], plot=plot)
        for p in range(0, an.ndim):
            R[p] = {'mean': dcoef_r[p, :].mean(),
                    'std': dcoef_r[p, :].std(),
                    'min': dcoef_r[p, :].min()}
        return R

    def explore(self, index, bounds, nu=None, nvals=51):
        """Explore the eigenvalues using brute force evaluation for one parameter.

        The CharPol is converted to a single parameter CharPol and its eigenvalues
        are evaluated inside the _absolute_ bounds.

        Parameters
        ----------
        index: int
            The index of the parameter to set.
        bounds: tuple
            The lower left and upper-right corner of the parameter
        nu: np.array, optional
            The value of the parameter vector. If `None` nu0 is used. The size
            of `nu` should be the same as the size of `nu0`, although The `nu[index]`
            is not used.
        nvals: int, optional
            The number of points in each complex plane direction.

        Returns
        -------
        loci: ee.Loci
            An instance of `Loci` containing all computation results.
        """
        if nu is None:
            nu = self.nu0
        # Create complex plane, bounds are absolute
        Re, Im = np.meshgrid(np.linspace(bounds[0].real, bounds[1].real, nvals),
                             np.linspace(bounds[0].imag, bounds[1].imag, nvals))
        Nu = Re + 1j * Im
        LAMBDA_pcp = np.zeros(Nu.shape, dtype=object)
        for item, nui in tqdm.tqdm(np.ndenumerate(Nu), total=np.prod(Nu.shape)):
            nu_ = nu.copy()
            nu_[index] = nui
            # Run the eigenvalue computation
            Lambda_ = self.eval_lda_at(nu_)
            # create a list of the eigenvalue to monitor
            LAMBDA_pcp[item] = Lambda_.copy()
        # Create Loci instance
        LAMBDA_pcp = np.array(LAMBDA_pcp.tolist())
        loci_pcp = Loci(LAMBDA_pcp, Nu)
        return loci_pcp


class Discr(Taylor):
    """Numerical representation of a Taylor expansion of the discriminant.
    """

    def locate(self, plot=False):
        """Find EP locations for univariate case only.

        Parameters
        ----------
        plot: bool
            If true, plot the roots of the discriminant compare to truncated version
            to identify the spurious roots.

        Returns
        -------
        r: array
            The roots represent all EP2 candodates sorted with descending sensitivity.
            The roots are express in an absolute way, ie nu0 is already included.
        s: array
            The sensitivity of each roots.
        """
        # Check the shape
        if len(self.nu0) > 1:
            raise ValueError(f'This method works only for univariate case. Found {len(self.nu0)}')
        # Use new numpy polynomial API
        p = pol.Polynomial(self.an)
        p1 = pol.Polynomial(self.an[:-1])
        r = p.roots()
        dp1 = p1.deriv()
        s = np.zeros_like(r)
        for i, ri in enumerate(r):
            s[i] = abs(p1(ri)/dp1(ri))
        # Sort the roots from the most genuine to the most spurious
        index = np.argsort(s)
        r = r[index] + self.nu0
        if plot:
            r1 = p1.roots() + self.nu0
            plt.figure()
            variable = r'\nu'
            plt.plot(r.real, r.imag, 'k.')
            plt.plot(r1.real, r1.imag, 'bo', markerfacecolor='none')
            plt.xlabel(r'$\mathrm{Re}\,' + variable + r' $')
            plt.ylabel(r'$\mathrm{Im}\,' + variable + r' $')
        return r, s
