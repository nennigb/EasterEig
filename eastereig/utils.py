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

""" Contains helper functions (combinatoric, ...)
"""
import numpy as np
from numpy import zeros, asarray, eye, poly1d, hstack, r_
from scipy import linalg
import itertools  as it

def multinomial_index_coefficients(m, n):
    r"""Return a tuple containing pairs ``((k1,k2,..,km) , C_kn)``
    where ``C_kn`` are multinomial coefficients such that
    ``n=k1+k2+..+km``.

    (x_1 + ... + x_m)**n

    Adapted from sympy sympy/ntheory/multinomial.py to return sorted index (speed up)
    to be sure that the first index is changing slowly

    Parameters:
    -----------
    m : int
        Number of variable
    n: int
        Power

    Returns
    --------
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


def multinomial_multiindex_coefficients(m, N):
    """ Compute the multinomial coefficients and indexes for multi-index.

    In multivariable Liebnitz formula to compute the derivatives as
    in (f*g*h)(x, y)^(N), where N is a tuple containing the order of derivation for each variable.

    It returns 2 lists that contains :
      - sublist with derivation order for all functions for all variable,
        for 2 functions f ang :
        [[(0, 0), (3, 5)], [(0, 1), (3, 4)],...]
        means [(df/dx0, dfdy0), (dg/dx3, dgdy5)], [(df/dx0, dfdy1), (dg/dx3, dgdy4)], ... ]
      - The product of the multinomial coefficients.

    Parameters:
    -----------
    m : integer
        Number of functions (same for all function)
    N: tuple
        contains the integer (Power, derivative order) for all variables.

    Returns
    --------
    multi_multi_index : list
        Containing all tuple (n1, n1, .., nm).
    multi_multi_coef : list
        Containing all coefficients.

    Examples
    --------
    This example has been tested with sympy
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
    """
    Return a list the sorted keys and the associated list of value
    """
    keys = adict.keys()
    sorted_keys = sorted(keys)
    return sorted_keys, [adict[key] for key in sorted_keys]


def pade(an, m, n=None):
    """
    Return Pade approximation to a polynomial as the ratio of two polynomials.
    version coming from scipy 1.4, previous version doesn't support complex
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
    >>> p(1)/q(1)
    (2.7179487179487...+0j)

    Compute Taylor exp(1+1j+x) @ x=0
    >>> e_expc = np.array([1.4686939399158851  +2.2873552871788423j, 1.4686939399158851  +2.2873552871788423j, \
                           0.7343469699579426  +1.1436776435894211j, 0.24478232331931418 +0.3812258811964737j, \
                           0.061195580829828546+0.09530647029911843j , 0.01223911616596571 +0.019061294059823684j])
    >>> p, q = pade(e_expc, 2)
    >>> p(-1j)/q(-1j)
    (2.7186371354862...+6.211398939416...e-05j)
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
