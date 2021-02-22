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

""" Contains helper functions (combinatoric, derivation, ...)
"""
import numpy as np
from numpy import zeros, asarray, eye, poly1d, hstack, r_
from scipy import linalg
from scipy.special import factorial, binom
import itertools as it
from collections import deque, namedtuple
from functools import reduce


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

    It appear in multivariable Liebnitz formula to compute derivatives like
    in (f*g*h)(x, y)^(N), where N is a tuple containing the order of derivation
    for each variable ('x' and 'y' here).

    It returns 2 lists that contains :
      - sublist with derivation order for all functions, for all variables,
        for instance, with 2 functions f and g such (f*g)(x, y)^(2, 3)
        [[(0, 0), (3, 5)], [(0, 1), (3, 4)],...]
        means [(df/dx0, dfdy0), (dg/dx3, dgdy5)], [(df/dx0, dfdy1), (dg/dx3, dgdy4)], ... ]
      - The product of the multinomial coefficients.

    Parameters:
    -----------
    m : integer
        Number of functions.
    N: tuple
        Contains the integer (derivative order) for all variables.

    Returns
    --------
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
    """ Return a list the sorted keys and the associated list of value.
    """
    keys = adict.keys()
    sorted_keys = sorted(keys)
    return sorted_keys, [adict[key] for key in sorted_keys]

# TODO : depreciated
def diffprod(dh, N):
    r"""Compute the n-th derivative of a product of function hi knowing the hi**(k).
    depending on a single variable.

    For instance, if H(x) = h0*h1*h2*...*h_(M-1) we want to compute
    H**(n) = (h0*h1*h2*...)**(n) with n<N

    This function use the generalized Liebnitz rule.

    Parameters
    ----------
    dh : list
        list of derivatives of all hi. The lenth of dh is M. Each element of dh
        contains the successive derivative of hi.
    N : int
        Number of requested derivatives

    Returns
    --------
    DH : list
        Contains the successive derivative of H. It is noteworthy that factorial
        are not included. DH is not a Taylor series.

    Exemples
    --------
    Let us consider a product of 2 function such,
    H = h0 * h1 with h0 = x**2, h1 = exp(x) @ x=1
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
    for n in np.arange(1, N):
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
            for t in np.arange(0, M):
                ph *= dh[t][k[t]]
            sh += ph*coefk
        # store nth derivatibe
        DH.append(sh)

    # DH contains the successive derivative, no factorial inside !!
    return DH


def diffprodMV(dh, N):
    r"""Compute the n-th derivative of a product of function hi knowing the
    hi**(k), depending on a single variable (if N is an int) or of multiple
    variables (N is a tuple).

    For instance, if H(x) = h0*h1*h2*...*h_(M-1) we want to compute
    H**(n) = (h0*h1*h2*...)**(n) with n<N

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
    --------
    DH : list
        Contains the successive derivatives of H with respect to all the variable.
        The output is ndarray whom dimensions follow N order. It is noteworthy that
        'factorial' are not included. DH is not a Taylor series.

    Exemples
    --------
    Multivariate example are available in the 'tests' folder.
    Let us consider H = h0 * h1 with h0 = x**2, h1 = exp(x) @ x=1
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
    
    """
    # TODO check type of N
    # Get the number of functions hi
    M = len(dh)

    # Check if single or multivariable using N
    mv = len(N) != 1

    # init DH of all h_i (no derivative)
    DH = np.zeros(np.array(N)+1, dtype=np.complex)
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
            for t in np.arange(0, M):
                ph *= dh[t][k[t]]
            sh += ph*coefk
        # store nth derivatibe
        DH[n] = sh

    # DH contains the successive derivative, no factorial inside !!
    return DH


def diffprodTree(dh, N):
    r"""Compute the n-th derivative of a product of function hi knowing the
    hi**(k), depending on a single variable (if N is an int) or of multiple
    variables (N is a tuple).

    For instance, if H(x) = h0*h1*h2*...*h_(M-1) we want to compute
    H**(n) = (h0*h1*h2*...)**(n) with n<N

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
    --------
    DH : list
        Contains the successive derivatives of H with respect to all the variable.
        The output is ndarray whom dimensions follow N order. It is noteworthy that
        'factorial' are not included. DH is not a Taylor series.

    Exemples
    --------
    Multivariate example are available in the 'tests' folder.
    Let us consider H = h0 * h1 with h0 = x**2, h1 = exp(x) @ x=1
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
    """ Convert Multivariate derivation matrix to Taylor series by dividing
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
    """ Compute the outer product of sequence of vectors.

    https://stackoverflow.com/questions/17138393/numpy-outer-product-of-n-vectors
    """
    return np.multiply.reduce(np.ix_(*vs))


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
    >>> p(1)/q(1)  # doctest: +ELLIPSIS
    (2.7179487179487...+0j)

    Compute Taylor exp(1+1j+x) @ x=0
    >>> e_expc = np.array([1.4686939399158851  +2.2873552871788423j, 1.4686939399158851  +2.2873552871788423j, \
                           0.7343469699579426  +1.1436776435894211j, 0.24478232331931418 +0.3812258811964737j, \
                           0.061195580829828546+0.09530647029911843j , 0.01223911616596571 +0.019061294059823684j])
    >>> p, q = pade(e_expc, 2)
    >>> p(-1j)/q(-1j)  # doctest: +ELLIPSIS
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


# Define named tupled to manipulate Bell polynomials
Bell = namedtuple('Bell', 'len coef pow')

def _partialBellPoly(N, K):
    r""" Compute partial Bell polynomials.

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
    B = np.empty((N+1, N+1), dtype=np.object)

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
    r""" Compute the successive derivative of a composite function with
    Faa Di Bruno formula.

    Expressed in terms of Bell polynomials Bn,k(x1,...,xn−k+1), this yields 
    \[ d^{n} \over dx^{n}}f(g(x))=\sum _{k=1}^{n}f^{(k)}(g(x))\cdot B_{n,k}\left(g'(x),g''(x),\dots ,g^{(n-k+1)}(x)\right). \]
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
    Compute the 4 first derivatives of the composite function exp(x)**2 at x=1.
    Let be f = g**2 and g(x) = exp(x).
    Define `df` the successive derivative of f with respect to g.
    Define `dg` the (constant) successive derivative of g with respect to x.
    >>> ref = np.array([7.38905609893065, 14.7781121978613, 29.5562243957226, 59.1124487914452, 118.224897582890])
    >>> x = 1.
    >>> dg = np.repeat(np.exp(x), 5)
    >>> df = np.array([np.exp(x)**2, 2*np.exp(x), 2, 0, 0])
    >>> dfog = faaDiBruno(df, dg)
    >>> np.linalg.norm(dfog - ref) < 1e-12
    True

    Compute the first derivatives of the composite function (x**3)**2 @ x=2
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
# %% Main for basic tests
if __name__ == '__main__':
    # run doctest Examples
    import doctest
    doctest.testmod()