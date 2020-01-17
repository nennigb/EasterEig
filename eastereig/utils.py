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
Contains helping functions (combinatoric, ...)
"""
import numpy as np
from numpy import zeros, asarray, eye, poly1d, hstack, r_
from scipy import linalg

def multinomial_index_coefficients(m, n):
    r"""Return a tuple containing pairs ``((k1,k2,..,km) , C_kn)``
    where ``C_kn`` are multinomial coefficients such that
    ``n=k1+k2+..+km``.
    
    Adapted from sympy sympy/ntheory/multinomial.py to return sorted index (speed up) 
    to be sure that the first index is changing slowly

    Returns
    --------
        mindex : 
        mcoef :
    
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


def sortdict(adict):
    """
    Return a list the sorted keys and the associated list of value
    """
    keys = adict.keys()
    sorted_keys = sorted(keys)
    return sorted_keys,[adict[key] for key in sorted_keys]


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
