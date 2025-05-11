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

"""Contains helper functions (combinatoric, derivation, ...)
"""
import numpy as np
from numpy import zeros, asarray, eye, poly1d, hstack, r_
from scipy import linalg
from scipy.special import factorial, binom
import itertools as it
from collections import deque, namedtuple
from functools import reduce
from functools import lru_cache
from eastereig.fpoly import polyvalnd
# Maxsize of the lru_cache
MAXSIZE = None


# lru_cache doesn't work out-of-the-box for generator
def two_composition(order, max_order):
    r"""Yield all the 2-compostion of `order` with list of two integers.

    Whereas for _partition_, the order matter. This combinatoric generator
    could be used to get all terms of a given order in the multiplcation of
    two polynomials.

    See https://gist.github.com/jasonmc/989158/682cfe1c25d39d5526acaeacc1426d287ef4f5de
    or https://tel.archives-ouvertes.fr/tel-00668134/document for generalization
    to k-composition of `order`.

    Parameters
    ----------
    order : int
        The targeted sum.
    max_order : iterable
        The max. value for the two items of the list.

    Yields
    ------
    tuple
        The composition members.

    Examples
    --------
    All possible arrangements to get 4 with two integers smaller than 4 and 3.
    >>> for c in two_composition(4, (4, 3)):
    ...     print(c, sum(c))
    (4, 0) 4
    (3, 1) 4
    (2, 2) 4
    (1, 3) 4

    Check that all members are there. The compositions of `s` into exactly `n`
    parts is given by the binomial coefficient \( {s+n-1 \choose s} \)
    see https://en.wikipedia.org/wiki/Composition_(combinatorics)
    >>> comp = list(two_composition(3, (3, 3)))
    >>> len(comp) == binom(3+2-1, 3)
    True
    """
    # Check if it is possible
    if order > sum(max_order):
        raise ValueError('Impossible to reach `order` from this `max_order`.')

    # Define the start state
    if order > max_order[0]:
        start = (max_order[0], order - max_order[0])
    else:
        start = (order, 0)

    # Define the stop criterion
    delta = order - max_order[1]
    stop = -1 if delta <= 0 else (delta-1)
    # Iteration loop
    for i in range(start[0], stop, -1):
        j = order-i
        yield (i, j)


@lru_cache(maxsize=MAXSIZE)
def multinomial_index_coefficients(m, n):
    r"""Return a tuple containing pairs ``((k1,k2,..,km) , C_kn)``
    where ``C_kn`` are multinomial coefficients such that
    ``n=k1+k2+..+km``.

    (x_1 + ... + x_m)**n

    Adapted from sympy sympy/ntheory/multinomial.py to return sorted index (speed up)
    to be sure that the first index is changing slowly

    Parameters
    ----------
    m : int
        Number of variable
    n: int
        Power

    Returns
    -------
    mindex : list
        Containing all tuple (k1,k2,..,km).
    mcoef : list
        Containing all coefficients.

    Examples
    --------
    >>> mindex, mcoef = multinomial_index_coefficients(2, 5)
    >>> mindex
    [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)]
    >>> mcoef
    [1, 5, 10, 10, 5, 1]

    Notes
    -----
    The algorithm is based on the following result:
    .. math::
        \binom{n}{k_1, \ldots, k_m} =
        \frac{k_1 + 1}{n - k_1} \sum_{i=2}^m \binom{n}{k_1 + 1, \ldots, k_i - 1, \ldots}
    Code contributed to Sage by Yann Laigle-Chapuy, copied with permission
    of the author.

    """
    m = int(m)
    n = int(n)
    if not m:
        if n:
            return {}
        return {(): 1}
    # FIXME check bounds !
    #    if m == 2:
    #        return binomial_coefficients(n)
    #    if m >= 2*n and n > 1:
    #        return dict(multinomial_coefficients_iterator(m, n))
    t = [n] + [0] * (m - 1)
    r = {tuple(t): 1}
    if n:
        j = 0  # j will be the leftmost nonzero position
    else:
        j = m
    # enumerate tuples in co-lex order
    while j < m - 1:
        # compute next tuple
        tj = t[j]
        if j:
            t[j] = 0
            t[0] = tj
        if tj > 1:
            t[j + 1] += 1
            j = 0
            start = 1
            v = 0
        else:
            j += 1
            start = j + 1
            v = r[tuple(t)]
            t[j] += 1
        # compute the value
        # NB: the initialization of v was done above
        for k in range(start, m):
            if t[k]:
                t[k] -= 1
                v += r[tuple(t)]
                t[k] += 1
        t[0] -= 1
        r[tuple(t)] = (v * tj) // (n - t[0])

    # sort the output to be sure that the first index is changing slowly
    mindex, mcoef = sortdict(r)
    return (mindex, mcoef)

@lru_cache(maxsize=MAXSIZE)
def multinomial_multiindex_coefficients(m, N):
    """Compute the multinomial coefficients and indexes for multi-index.

    It appear in multivariable Liebnitz formula to compute derivatives like
    in (f*g*h)(x, y)^(N), where N is a tuple containing the order of derivation
    for each variable ('x' and 'y' here).

    It returns 2 lists that contains :
      - sublist with derivation order for all functions, for all variables,
        for instance, with 2 functions f and g such (f*g)(x, y)^(2, 3)
        [[(0, 0), (3, 5)], [(0, 1), (3, 4)],...]
        means [(df/dx0, dfdy0), (dg/dx3, dgdy5)], [(df/dx0, dfdy1), (dg/dx3, dgdy4)], ... ]
      - The product of the multinomial coefficients.

    Parameters
    ----------
    m : integer
        Number of functions.
    N: tuple
        Contains the integer (derivative order) for all variables.

    Returns
    -------
    multi_multi_index : list
        Containing all tuple (n1, n1, .., n_m).
    multi_multi_coef : list
        Containing the product of the multinomial coefficients.

    Examples
    --------
    This example has been tested with sympy (see 'tests' folder).
    >>> mindex, mcoef = multinomial_multiindex_coefficients(2, (2, 3))
    >>> mindex
    [[(0, 0), (2, 3)], [(0, 1), (2, 2)], [(0, 2), (2, 1)], [(0, 3), (2, 0)], [(1, 0), (1, 3)], \
[(1, 1), (1, 2)], [(1, 2), (1, 1)], [(1, 3), (1, 0)], [(2, 0), (0, 3)], [(2, 1), (0, 2)], \
[(2, 2), (0, 1)], [(2, 3), (0, 0)]]
    >>> mcoef
    [1, 3, 3, 1, 2, 6, 6, 2, 1, 3, 3, 1]
    """
    # Create the multinomial index/coef for each variable
    mindex, mcoef = [], []
    for n in N:
        mindex_, mcoef_ = multinomial_index_coefficients(m, n)
        mindex.append(mindex_)
        mcoef.append(mcoef_)

    # Create the gloabl index/coef
    multi_multi_index = []
    multi_multi_coef = []
    for mmindexi, mmcoefi in zip(it.product(*mindex),
                                 it.product(*mcoef)):
        # gloabl coef are the product of each coefs
        multi_multi_coef.append(np.prod(mmcoefi))
        # rearrange data structure to regroup them by function
        multi_multi_index.append([pair for pair in zip(*mmindexi)])

    return (multi_multi_index, multi_multi_coef)


def sortdict(adict):
    """Return a list the sorted keys and the associated list of value.
    """
    keys = adict.keys()
    sorted_keys = sorted(keys)
    return sorted_keys, [adict[key] for key in sorted_keys]


# TODO : depreciated
def diffprod(dh, N):
    r"""Compute the n-th derivative of a product of function hi knowing the hi**(k)
    depending on a single variable.

    For instance, if H(x) = h0*h1*h2*...*h_(M-1) we want to compute
    H^(n) = (h0*h1*h2*...)^(n) with n<N

    This function use the generalized Liebnitz rule.

    Parameters
    ----------
    dh : list
        list of derivatives of all hi. The lenth of dh is M. Each element of dh
        contains the successive derivative of hi.
    N : int
        Number of requested derivatives

    Returns
    -------
    DH : list
        Contains the successive derivative of H. It is noteworthy that factorial
        are not included. DH is not a Taylor series.

    Exemples
    --------
    Let us consider a product of 2 function such,
    H = h0 * h1 with h0 = x^2, h1 = exp(x) @ x=1
    >>> dh = [np.array([1, 2*1, 2, 0, 0]), np.exp(1)*np.ones((5,))]
    >>> dh_ref = np.array([2.71828182845905, 8.15484548537714, 19.0279727992133, 35.3376637699676])
    >>> d = diffprod(dh, 4)
    >>> np.linalg.norm(dh_ref - np.array(d)) < 1e-10
    True

    # With a product of 3 functions : h0 = x, h1 = exp(x), h2=x @ x=1
    >>> dh = [np.array([1, 1, 0, 0, 0]), np.exp(1)*np.ones((5,)), np.array([1, 1, 0, 0, 0])]
    >>> d = diffprod(dh, 4)
    >>> np.linalg.norm(dh_ref - np.array(d)) < 1e-10
    True
    """
    # Get the number of functions hi
    M = len(dh)

    # init DH of all h_i (no derivative)
    DH = [np.prod([dh[i][0] for i in range(0, M)])]
    # compute global derivative order n
    for n in range(1, N):
        # get multinomial index and coefficients
        multi, coef = multinomial_index_coefficients(M, n)
        # liebnitz formula
        # sum
        sh = complex(0.)
        for index, k in enumerate(multi):
            # coefk = multinom(n,k)
            coefk = coef[index]
            # produit
            ph = complex(1.)
            for t in range(0, M):
                ph *= dh[t][k[t]]
            sh += ph*coefk
        # store nth derivatibe
        DH.append(sh)

    # DH contains the successive derivative, no factorial inside !!
    return DH


def diffprodMV(dh, N):
    r"""Compute the n-th derivative of a product of function hi knowing the
    hi^(k), depending on a single variable (if N is an int) or of multiple
    variables (N is a tuple).

    For instance, if H(x) = h0*h1*h2*...*h_(M-1) we want to compute
    H^(n) = (h0*h1*h2*...)^(n) with n<N

    This function use the generalized Liebnitz rule.

    Parameters
    ----------
    dh : list
        ndarray of derivatives of all hi. The lenth of dh is M. Each element of dh
        contains the successive derivative of hi in a ndarray.
    N : int or tuple
        Number of requested derivatives. For uni-varaite case, N must be an integader,
        for the multi-variate case, N must be tuple containing the derivation order
        for each variable.

    Returns
    -------
    DH : list
        Contains the successive derivatives of H with respect to all the variable.
        The output is ndarray whom dimensions follow N order. It is noteworthy that
        'factorial' are not included. DH is not a Taylor series.

    Exemples
    --------
    Multivariate example are available in the 'tests' folder.
    Let us consider H = h0 * h1 with h0 = x^2, h1 = exp(x) @ x=1
    >>> dh = [np.array([1, 2*1, 2, 0, 0]), np.exp(1)*np.ones((5,))]
    >>> dh_ref = np.array([2.71828182845905, 8.15484548537714, 19.0279727992133, 35.3376637699676])
    >>> d = diffprodMV(dh, (3,))
    >>> np.linalg.norm(dh_ref - np.array(d)) < 1e-10
    True

    # With 3 functions : h0 = x, h1 = exp(x), h2=x @ x=1
    >>> dh = [np.array([1, 1, 0, 0, 0]), np.exp(1)*np.ones((5,)), np.array([1, 1, 0, 0, 0])]
    >>> d = diffprodMV(dh, (3,))
    >>> np.linalg.norm(dh_ref - np.array(d)) < 1e-10
    True

    Other examples are present in `test_multiindex.py`.
    """
    # TODO check type of N
    # Get the number of functions hi
    M = len(dh)

    # Check if single or multivariable using N
    mv = len(N) != 1

    # init DH of all h_i (no derivative)
    DH = np.zeros(np.array(N)+1, dtype=complex)
    # Create the generator
    if mv:
        # for multivariable case
        derivative_deg = it.product(*map(range, np.array(N)+1))
        multinomial = multinomial_multiindex_coefficients
    else:
        # for mono variable case
        derivative_deg = np.arange(0, N[0]+1)
        multinomial = multinomial_index_coefficients

    # Compute global derivative order n
    for n in derivative_deg:
        # Get multinomial index and coefficients
        multi, coef = multinomial(M, n)
        # Liebnitz formula
        # sum
        sh = complex(0.)
        for index, k in enumerate(multi):
            # coefk = multinom(n,k)
            coefk = coef[index]
            # produit
            ph = complex(1.)
            for t in range(0, M): # TODO try to improve more cython, jit...
                ph *= dh[t][k[t]]
            sh += ph*coefk
        # store nth derivatibe
        DH[n] = sh

    # DH contains the successive derivative, no factorial inside !!
    return DH

def diffprodTree(dh, N):
    r"""Compute the n-th derivative of a product of function hi knowing the
    hi^(k), depending on a single variable (if N is an int) or of multiple
    variables (N is a tuple).

    For instance, if H(x) = h0*h1*h2*...*h_(M-1) we want to compute
    H^(n) = (h0*h1*h2*...)^(n) with n<N

    This function use the generalized Liebnitz rule by pair using a queue.
    This approach is faster that `diffprod` when the number of function is
    more than 3.

    Parameters
    ----------
    dh : list
        ndarray of derivatives of all hi. The lenth of dh is M. Each element of dh
        contains the successive derivative of hi in a ndarray.
    N : int or tuple
        Number of requested derivatives. For uni-varaite case, N must be an integader,
        for the multi-variate case, N must be tuple containing the derivation order
        for each variable.

    Returns
    -------
    DH : list
        Contains the successive derivatives of H with respect to all the variable.
        The output is ndarray whom dimensions follow N order. It is noteworthy that
        'factorial' are not included. DH is not a Taylor series.

    Exemples
    --------
    Multivariate example are available in the 'tests' folder.
    Let us consider H = h0 * h1 with h0 = x^2, h1 = exp(x) @ x=1
    >>> dh = [np.array([1, 2*1, 2, 0, 0]), np.exp(1)*np.ones((5,))]
    >>> dh_ref = np.array([2.71828182845905, 8.15484548537714, 19.0279727992133, 35.3376637699676])
    >>> d = diffprodTree(dh, (3,))
    >>> np.linalg.norm(dh_ref - np.array(d)) < 1e-10
    True

    # With 3 functions : h0 = x, h1 = exp(x), h2=x @ x=1
    >>> dh = [np.array([1, 1, 0, 0, 0]), np.exp(1)*np.ones((5,)), np.array([1, 1, 0, 0, 0])]
    >>> d = diffprodTree(dh, (3,))
    >>> np.linalg.norm(dh_ref - np.array(d)) < 1e-10
    True

    """
    # create a queue to be consumed
    lda_list = deque(range(0, len(dh)))
    # create a dictionnary containing all the derivatives of the tree
    deriv = dict(zip(lda_list, dh))

    # consume the queue by computing derivative of successive pairs
    while len(lda_list) > 1:
        # get the pair
        pair = (lda_list.popleft(), lda_list.popleft())
        lda_list.append(pair)
        # store the results in the dict
        # FIXME don't forget to change mane here
        deriv[pair] = diffprodMV((deriv[pair[0]], deriv[pair[1]]), N)

    # return the final derivative
    return deriv[lda_list[0]]


# Nothing to do, juste create an alias (just in case)
diffprodTreeMV = diffprodTree

def div_factorial(dH):
    """Convert Multivariate derivation matrix to Taylor series by dividing
    by the factorial coeff.

    Parameters
    ----------
    dH : ndarray
        Contains the derivative of H with respect of each variable.
        dH[3,2] means up dx**2 dy (H).

    # TODO chnage into factorial matrix to avoid to recomputed...
    Returns
    -------
    Th : ndarray
        Taylor series coefficients.
    """
    N = dH.shape
    nmax = max(N)
    # compute the factorial once.
    fact = factorial(np.arange(0, nmax))
    fact_list = [fact[0:ni] for ni in N]
    D = _outer(*fact_list)
    Th = dH/D
    return Th


def _outer(*vs):
    """Compute the outer product of sequence of vectors.

    https://stackoverflow.com/questions/17138393/numpy-outer-product-of-n-vectors

    Parameters
    ----------
    vs : tuple of iterable
        Each element is an array that will contribute to the outer product.
    """
    return reduce(np.multiply.outer, vs)


def pade(an, m, n=None):
    """
    Return Pade approximation to a polynomial as the ratio of two polynomials.

    Version coming from scipy 1.4, previous version doesn't support complex
    with modify test case to check complex behaviour
    [scipy.interpolate](https://github.com/scipy/scipy/blob/master/scipy/interpolate/_pade.py)

    Parameters
    ----------
    an : (N,) array_like
        Taylor series coefficients in ascending order.
    m : int
        The order of the returned approximating polynomial `q`.
    n : int, optional
        The order of the returned approximating polynomial `p`. By default,
        the order is ``len(an)-m``.

    Returns
    -------
    p, q : Polynomial class
        The Pade approximation of the polynomial defined by `an` is
        ``p(x)/q(x)``.

    Examples
    --------
    adapt to complex
    >>> e_exp = [1.0+0j, 1.0+0.j, (1.0+0.j)/2.0, (1.0+0.j)/6.0, (1.0+0.j)/24.0, (1.0+0.j)/120.0]
    >>> p, q = pade(e_exp, 2)
    >>> p(1)/q(1)  # doctest: +ELLIPSIS
    (2.7179487179487...+0j)

    Compute Taylor exp(1+1j+x) @ x=0
    >>> e_expc = np.array([1.4686939399158851  +2.2873552871788423j, 1.4686939399158851  +2.2873552871788423j, \
                           0.7343469699579426  +1.1436776435894211j, 0.24478232331931418 +0.3812258811964737j, \
                           0.061195580829828546+0.09530647029911843j , 0.01223911616596571 +0.019061294059823684j])
    >>> p, q = pade(e_expc, 2)
    >>> p(-1j)/q(-1j)  # doctest: +ELLIPSIS
    (2.7186371354...+6.2113989394...e-05j)
    """
    an = asarray(an)
    if n is None:
        n = len(an) - 1 - m
        if n < 0:
            raise ValueError("Order of q <m> must be smaller than len(an)-1.")
    if n < 0:
        raise ValueError("Order of p <n> must be greater than 0.")
    N = m + n
    if N > len(an)-1:
        raise ValueError("Order of q+p <m+n> must be smaller than len(an).")
    an = an[:N+1]
    Akj = eye(N+1, n+1, dtype=an.dtype)
    Bkj = zeros((N+1, m), dtype=an.dtype)
    for row in range(1, m+1):
        Bkj[row, :row] = -(an[:row])[::-1]
    for row in range(m+1, N+1):
        Bkj[row, :] = -(an[row-m:row])[::-1]
    C = hstack((Akj, Bkj))
    pq = linalg.solve(C, an)
    p = pq[:n+1]
    q = r_[1.0, pq[n+1:]]
    return poly1d(p[::-1]), poly1d(q[::-1])


# Define named tupled to manipulate Bell polynomials
Bell = namedtuple('Bell', 'len coef pow')


def _partialBellPoly(N, K):
    r"""Compute partial Bell polynomials.

    These polynomials appear in the computation of derivatives of composite function.

    The exponential Bell polynomial encodes the information related to the
    ways a set of N can be partitioned in K subsets.

    The partial Bell polynomials are computed by a recursion relation
    (https://en.wikipedia.org/wiki/Bell_polynomials#Recurrence_relations) :
    $$ B_{n,k}=\sum _{i=1}^{n-k+1}{\binom {n-1}{i-1}}x_{i}B_{n-i,k-1} $$
    where \( B_{0,0}=1\), \( B_{n,0}=0\;\mathrm {for} \;n\geq 1\) and
    \(B_{0,k}=0\;\mathrm {for} \;k\geq 1.\)

    Parameters
    ----------
    N : int
        Set size.
    K : int
        subset size.

    Returns
    -------
    B : array
        return all the partial Bell polynomial. Each polynomial is represented as a named-tuple,
        with len, pow and coef field. `coef[i]` constains the coefficient of the i-thmonomial
        and `pow[i]` contains the expononent of each symbolic variable `x_i`
        (see below in the example).

    Examples
    --------
    For instance, to compute
    $$ B_{6,3}(x_{1},x_{2},x_{3},x_{4})= 15x_{4}x_{1}^{2}+60x_{3}x_{2}x_{1}+15x_{2}^{3} $$
    >>> B = _partialBellPoly(6, 3)
    >>> B[6, 3]  # doctest: +NORMALIZE_WHITESPACE
    Bell(len=3,
         coef=array([15, 60, 15], dtype=int32),
         pow=array([[0, 3, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [2, 0, 0, 1, 0, 0]], dtype=int32))

    Alternativelly, such number can be computed in sympy, using
    `bell(6, 3, symbols('x:6')[1:])`. It means there are :
      * 15 ways to partition a set of 6 as 4 + 1 + 1, -> x4 x1**2
      * 60 ways to partition a set of 6 as 3 + 2 + 1, and -> x3 x2 x1
      * 15 ways to partition a set of 6 as 2 + 2 + 2.   -> x2**3

    \[B_{6,2} = 6 x_{1} x_{5} + 15 x_{2} x_{4} + 10 x_{3}^{2}￼ \]
    >>> B[6, 2]  # doctest: +NORMALIZE_WHITESPACE
    Bell(len=3,
         coef=array([10, 15,  6], dtype=int32),
         pow=array([[0, 0, 2, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [1, 0, 0, 0, 1, 0]], dtype=int32))
    """
    # Init output results
    B = np.empty((N+1, N+1), dtype=object)

    # Create 0 bell polynomials
    b0 = Bell(0, np.array([0]), np.zeros((1, N), dtype=np.int32))
    # Recursive building
    for n in range(0, N+1):
        for k in range(0, n+1):
            if (k == 0) and (n == 0):
                # B[0, 0] = 1
                B[0, 0] = Bell(1, np.array([1]), np.zeros((1, N), dtype=np.int32))
            elif ((k >= 1) and (n == 0)) or ((n >= 1) and (k == 0)):
                # case k==0, fill with b0
                B[n, k] = b0
            else:
                # Compute B(n,k)
                # Inititalize temp field
                coef = []
                Pow = []
                for i in range(1, n-k+2):
                    c = binom(n-1, i-1)
                    # may contains several terms
                    for j in range(1, B[n-i, k-1].len + 1):
                        # multiply with binomial coef
                        coef_ = B[n-i, k-1].coef[j-1]*c
                        pow_ = B[n-i, k-1].pow[j-1].copy()
                        # shift power
                        pow_[i-1] = pow_[i-1] + 1
                        # store them
                        Pow.append(pow_)
                        coef.append(coef_)

                # Find polynomials that appear several time, and sum there coeff
                # powu = pow(ic), ic previous size, but with new index
                powu, ia, ic = np.unique(Pow, axis=0, return_index=True, return_inverse=True)
                coefu = np.zeros(ia.size, dtype=np.int32)
                for j in range(0, len(coef)):
                    coefu[ic[j]] = coef[j] + coefu[ic[j]]

                # Store the new polynomial as Bell object
                B[n, k] = Bell(len(coefu), coefu, powu)

    return B


def faaDiBruno(df, dg, N=None):
    r"""Compute the successive derivative of a composite function with
    Faa Di Bruno formula.

    Expressed in terms of Bell polynomials Bn,k(x1,...,xn−k+1), this yields
    $${d^{n} \over dx^{n}}f(g(x))=\sum _{k=1}^{n}f^{(k)}(g(x))\cdot B_{n,k}\left(g'(x),g''(x),\dots ,g^{(n-k+1)}(x)\right). $$
    see https://en.wikipedia.org/wiki/Fa%C3%A0_di_Bruno's_formula

    Parameters
    ----------
    df : array
        value of df/dg computed at x0. By convention, df[0] is the function value
        (not the derivatives).
    dg : array
        values of dg/dx computed at x0. By convention, dg[0] is the function value
        (not the derivatives).
    N : int, optional
        The requested number of derivatives if less than `len(df)`

    Returns
    -------
    dfog : array
        Values of the successive derivative of the composite function. `len(dfog)` is N+1
        or `len(df)`.

    Examples
    --------
    Compute the 4 first derivatives of the composite function exp(x)^2 at x=1.
    Let be f = g^2 and g(x) = exp(x).
    Define `df` the successive derivative of f with respect to g.
    Define `dg` the (constant) successive derivative of g with respect to x.
    >>> ref = np.array([7.38905609893065, 14.7781121978613, 29.5562243957226, 59.1124487914452, 118.224897582890])
    >>> x = 1.
    >>> dg = np.repeat(np.exp(x), 5)
    >>> df = np.array([np.exp(x)**2, 2*np.exp(x), 2, 0, 0])
    >>> dfog = faaDiBruno(df, dg)
    >>> np.linalg.norm(dfog - ref) < 1e-12
    True

    Compute the first derivatives of the composite function (x^3^2 @ x=2
    >>> x = 2.
    >>> dg = np.array([x**3, 3*x**2, 6*x, 6, 0, 0, 0])
    >>> df = np.array([dg[0]**2, 2*dg[0], 2, 0, 0, 0, 0])
    >>> dfog = faaDiBruno(df, dg)
    >>> ref = np.array([64.0, 192.0, 480.0, 960.0, 1440.0, 1440.0, 720.0])
    >>> np.linalg.norm(dfog - ref) < 1e-12
    True
    """
    # Get the maximal number of available derivatives
    if (N is None):
        N = len(df) - 1
    # Get the maximal number of df non vanishing terms,
    # Remove trailling 0 to speed up loop.
    kmax = np.flatnonzero(df)[-1] + 1
    # Compute required Bell polynomials
    B = _partialBellPoly(N, kmax)
    # Compute the derivative up to N
    dfog = []
    for n in range(0, N+1):
        if n == 0:
            dfog.append(df[0])
        else:
            # init Bell polynomial storage variable
            dfog_ = 0j
            # Sum until df has trailling zeros
            for k, dfk in enumerate(df[0:min(n+1, kmax)]):
                if k > 0:
                    s = 0j
                    for i, c in enumerate(B[n, k].coef):
                        s += c * np.prod(np.power(dg[1:(n-k+2)],
                                                  B[n, k].pow[i][0:(n-k+1)]))
                    # multipy by the function f derivatives
                    dfog_ += dfk * s
            dfog.append(dfog_)

    return dfog


def complex_map(bounds=(-5-5j, 5+5j), N=30):
    """Create a map of the complex plane between the bounds.

    Parameters
    ----------
    bounds : tuple
        The two corners on the map in the complex plane.
    N : int
        The number of points in each direction.

    Returns
    -------
    np.array:
        The complex coordinate of the points.
    """
    zr, zi = np.meshgrid(np.linspace(bounds[0].real, bounds[1].real, N),
                         np.linspace(bounds[0].imag, bounds[1].imag, N))
    return zr + 1j*zi


class Taylor:
    r"""Define a multivariate Taylor series.

    The series is defined as
    \(T = \sum_{n_0, n_1, \dots} a_{n_0, n_1, \dots} \nu_0^{n_0}, \nu_1^{n_1} \dots\)
    where the constant 0-order term is \(a_{0, 0, \dots}\).
    """

    def __init__(self, an, nu0):
        """Initialize the object with df/d nu divided by factorial.

        Parameters
        ----------
        an : np.ndarray
            The coefficients of the Taylor expansion.
        nu0 : iterable
            The value where the Taylor series is computed.
        """
        self.an = an
        self.nu0 = nu0

    @classmethod
    def from_derivatives(cls, dn, nu0):
        """Instanciate Taylor object from the sucessive derivatives df/dnu.

        Parameters
        ----------
        dn : array
            The coefficients of the function derivatives wrt nu.
        nu0 : iterable
            The value where the derivatives are computed.

        Returns
        -------
        Taylor
            An new Taylor instance.
        """
        an = div_factorial(dn)
        return cls(an, nu0)

    def __repr__(self):
        """Define the representation of the class."""
        return "Instance of {} @nu0={} with #{} derivatives.".format(self.__class__.__name__,
                                                                     self.nu0,
                                                                     self.an.shape)

    def eval_at(self, nu):
        """Evaluate the Taylor series at nu with derivatives computed at nu0.

        Parameters
        ----------
        nu : iterable
            Containts the value of (nu_0, ..., nu_n) where the polynom
            must be evaluated. Althought nu is relative to nu0, absolute value
            have to be used.

        Returns
        -------
        The evaluation of the Taylor series at nu.

        Examples
        --------
        Works also for scalar case,
        >>> T = Taylor(1/factorial(np.arange(0, 6)), 0.)
        >>> np.allclose(T.eval_at(0.1), np.exp(0.1), atol=1e-4)
        True
        """
        # Extract input value
        nu = np.array(nu, dtype=complex) - np.array(self.nu0, dtype=complex)
        # Evaluate the polynomial
        return polyvalnd(nu, self.an)

    @staticmethod
    def _radii_fit_1D(a, plot=False):
        r"""Estimate the radii of convergence using least square of 1D sequence.

        Based on Cauchy-Hadamard,
        $$\frac{1}{\rho} = |a_n|^{1/n}$$
        yields to
        $$-\mathbf{V} \alpha + beta = \ln{\mathbf{a}}$$

        For instance if some terms are vanishing, like odd or even terms,
        the LS fit add a bias (The theory introduce the sup lim!). To limit
        this, approach based on peaks location may works (not implemented).

        Parameters
        ----------
        a: array 1D
            The Taylor series coefficients.
        plt: bool, optional
            If `True` plots the data and the fit.

        Return
        ------
        rho: float
            The radius of convergence.
        """
        # tol for detecting alternate vanishing coefs
        tol = 0.25
        eps = 1e-16
        # Remove log singularity
        a = a + eps
        D = a.size
        alp = np.arange(0, D)[:, None]
        # LS fit
        V = np.hstack((np.ones((D, 1)), alp))
        # beta correspond to the normalization by a0
        beta, alpha = np.linalg.pinv(V) @ (np.log(np.abs(a).reshape(-1, 1)))
        rho = np.exp(-alpha.flat[0])
        # Estimation based only on last term
        rho_ = 1/(np.abs((a[-1]/a[0]))**(1/(a.size-1)))
        if abs(a[0]) < eps:
            print('Warning: The first term is near zero.')
        if (rho-rho_)/rho > tol:
            print('Warning: suspect alternate vanishing coefficients rhof={}, rho1={}.'.format(rho, rho_))
        if plot:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.plot(alp, np.log(np.abs(a)), '*', label='True')
            # plt.plot(alp, np.log(np.abs(a_) + eps), '.', label='MV')
            plt.plot(alp, beta + alp * alpha, label='fit')
            plt.ylabel(r'$\log{a_i}$')
            plt.xlabel(r'Power')
            plt.legend()
        return rho

    def radii_estimate(self, plot=False):
        r"""Estimate the radii of convergence of the Taylor expansion.

        The approach is based on least square fit for all paremeters and
        all the series coefficients and assume single variable dependance.

        Returns
        -------
        R : array
            The radius of convergence for along all directions.
        plt: bool, optional
            If `True` plots the data and the fit.

        Examples
        --------
        If $$\frac {1}{1-x} = \sum _{n=0}^{\infty }x^{n}$$, with radius of convergence 1.
        >>> T = Taylor(np.ones((12,)), 0)
        >>> np.allclose(T.radii_estimate(), 1, atol=1e-4)
        True

        When test on $$\tan(x)$$ around 0, it fails because of alternate vanishing terms,
        illustrating the importance of sup lim in the theory,
        >>> T = Taylor(np.array([0, 1.00000000000000, 0, 0.333333333333333, 0, 0.133333333333333, 0, 0.0539682539682540, 0, 0.0218694885361552,0,  0.00886323552990220]), 0)
        >>> T.radii_estimate()  # doctest: +ELLIPSIS
        Warning: ...

        When test on $$\tan(1+x)$$ around 0, it works better
        >>> T = Taylor(np.array([3.42551882081476, 5.33492947248766, 9.45049997787964, 16.4965914915633, 28.9182083191928, 50.6548588382890]), 0)
        >>> np.allclose(T.radii_estimate(), np.pi/2 -1, atol=1e-1)
        True
        """
        an = self.an
        R = np.zeros(an.ndim)
        # Loop over the parameters
        for p in range(0, an.ndim):
            index = [0] * an.ndim
            index[p] = slice(0, None)
            R[p] = Taylor._radii_fit_1D(an[tuple(index)])
        return R


    # def myroots(coef):
    #     """Provide a global interface for roots of polynom with variable coefficients.

    #     The polynomial is defined such, coef[0] x**n + ... + coef[-1] and


    #     Analytic formula are used for qudratric and cubic polynomial and numerical
    #     methods are used for higher order.


    #     Parameters
    #     ----------
    #     coef : list of complex array like
    #         Contains the value of each coefficients.

    #     Returns
    #     -------
    #     r : list
    #         list of roots for each vector coef

    #     """

    #     # number of coef of the polynome
    #     n = len(coef)
    #     if n == 3:
    #         # quadratique
    #         r = quadratic_roots(*coef)
    #     elif n==4:
    #         # cubic
    #         r = cubic_roots(*coef)
    #     elif n==5:
    #         # quatic, only exceptionhandle is the Biquadratic equation
    #         r = quartic_roots(*coef)

    #     else:
    #         # quelquonque
    #         N = len(coef[0])
    #         r=np.zeros((n-1, N), dtype=complex)

    #         for i, c in enumerate(zip(*coef)):
    #             print(c)
    #             ri = np.roots(c)
    #             r[:,i] = ri

    #     return r


    # def quadratic_roots(a, b, c):
    #     """Analytic solution of a quadratic polynom P(x)=ax^2+bx+c.

    #     Parameters
    #     ----------
    #     a, b, c : complex array like
    #         coeff of the polynom such

    #     Returns
    #     --------
    #     r : list
    #         list of roots
    #     """
    #     tol = 1e-8
    #     # check on 1st coef
    #     if np.linalg.norm(a)< tol:
    #         raise ValueError('The first coefficient cannot be 0.')

    #     Delta = np.sqrt(b**2 - 4*a*c, dtype=np.complex)
    #     r1 = (- b - Delta)/(2*a)
    #     r2 = (- b + Delta)/(2*a)
    #     r = [r1, r2]

    #     return r

    # def cubic_roots(a,b,c,d):
    #     """Analytic solution of a cubic polynom  P(x)=ax^3+bx^2+cx+d.

    #     Parameters
    #     ----------
    #     a, b, c, d : complex array like
    #         coeff of the polynom such

    #     Returns
    #     --------
    #     r : list
    #         list of roots

    #     Examples
    #     --------
    #     Solve `x**3 - 4.0*x**2 + 6.0*x - 4.0` with complex roots,
    #     >>> coef = [1, -4, 6, -4]
    #     >>> r = np.array(cubic_roots(*coef)); r.sort()
    #     >>> r_ = np.array([2, 1+1j, 1-1j]); r_.sort()
    #     >>> np.linalg.norm(r - r_) < 1e-12
    #     True
    #     """
    #     tol = 1e-8
    #     # check on 1st coef
    #     if np.linalg.norm(a)< tol:
    #         raise ValueError('The first coefficient cannot be 0.')

    #     D0 = b**2 - 3*a*c
    #     D1 = 2*b**3 - 9*a*b*c + 27*(a**2)*d
    #     C = ((D1 + np.sqrt(D1**2 - 4*D0**3, dtype=np.complex))/2.)**(1./3.)
    #     xi = -0.5 + 0.5*np.sqrt(3)*1j
    #     x = []
    #     for k in range(1, 4):
    #         xk = -1/(3*a)*(b + xi**k*C + D0/(xi**k*C))
    #         x.append(xk)
    #     return x

    # def quartic_roots(a,b,c,d,e):
    #     """Analytic solution of a quartic polynom P(x)=ax^4+bx^3+cx^2+dx+e.

    #     Parameters
    #     ----------
    #     a, b, c, d, e : complex array like
    #         coeff of the polynom. Better to unpack list to set them.

    #     Returns
    #     --------
    #     r : array
    #         list of roots

    #     Remarks
    #     --------
    #     Only the biquadratic exception is implemented. Through extensive tests, no
    #     other problems have been found, but...
    #     https://en.wikipedia.org/wiki/Quartic_function

    #     Examples
    #     --------
    #     Solve `x**4 - 7.0*x**3 + 2.0j*x**3 + 18.0*x**2 - 8.0j*x**2 - 22.0*x + 12.0j*x + 12.0 - 8.0j` with complex roots,
    #     >>> coef = [1, -7.0 + 2.0j, 18.0 - 8.0j, -22.0 + 12.0j, 12.0 - 8.0j]
    #     >>> r = np.array(quartic_roots(*coef)); r.sort()
    #     >>> r_ = np.array([1+1j , 2, 1-1j, 3-2j]); r_.sort()
    #     >>> np.linalg.norm(r - r_) < 1e-12
    #     True

    #     Biquadratic equation (degenerate case)
    #     >>> coef = [1, 0, -9 + 2j, 0, -18j]
    #     >>> r = np.array(quartic_roots(*coef)); r.sort()
    #     >>> r_ = np.array([3, -3, 1-1j, -1+1j]); r_.sort()
    #     >>> np.linalg.norm(r - r_) < 1e-12
    #     True
    #     """
    #     tol = 1e-8
    #     # check on 1st coef
    #     if np.linalg.norm(a)< tol:
    #         raise ValueError('The first coefficient cannot be 0.')

    #     Delta0 = c**2 - 3*b*d + 12*a*e
    #     Delta1 = 2*c**3 - 9*b*c*d + 27*b**2*e + 27*a*d**2 - 72*a*c*e
    #     p = (8*a*c - 3*b**2)/(8*a**2)
    #     q = (b**3 - 4*a*b*c + 8*a**2*d) / (8*a**3)
    #     if np.linalg.norm(q)< tol:
    #         # degenerate case of Biquadratic equation
    #         r = quadratic_roots(*[a, c, e])
    #         r[0] = np.sqrt(r[0])
    #         r[1] = np.sqrt(r[1])
    #         r.extend([-r[0], -r[1]])
    #     else:
    #         Q = np.power( 0.5*(Delta1 + np.sqrt(Delta1**2 - 4*Delta0**3, dtype=np.complex)), 1/3., dtype=np.complex)
    #         S = 0.5*np.sqrt(-2*p/3 + 1/(3*a)*(Q + Delta0/Q), dtype=np.complex )


    #         B = -b/(4*a)
    #         Dp = 0.5*np.sqrt(-4*S**2 - 2*p + q/S, dtype=np.complex)
    #         Dm = 0.5*np.sqrt(-4*S**2 - 2*p - q/S, dtype=np.complex)
    #         r = [B - S + Dp,
    #              B - S - Dp,
    #              B + S + Dm,
    #              B + S - Dm]
    #     return r
# %% Main for basic tests
if __name__ == '__main__':
    # run doctest Examples
    import doctest
    doctest.testmod()
