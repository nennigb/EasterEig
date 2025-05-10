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

r""" Define the functions to compute the successive derivatives of the function of the eigenvalue
with respect to nu. These functions appear in
L=K_0 * None + K_1*Lda(nu) + K_2 * Lda2(nu) + ... + K_n * f_n(lda(nu))

  - These functions sould have the same interface : my_fun(k, n, dlda)
  - **If k==n, the terms containing lda^(n) are skipped**. They dont belong to the RHS computation
  - a companion function providing the derivitive with respect to lda should be also defined and
  enrolled in `dlda_flda` dict (at the end) define a dict to get the derivative % lda of the
  function of lda

Remarks:
--------
  * Only linear and quadratic dependancy are now implemented for multivariate case.
  * If functions are added, don't forget do update `_dlda_flda`.
  * `faaDiBruno` method allows to compute the derivative for a large variety of composite.
    function \( f(\lambda (\nu)) \) with scalar parameter (see example with `Lda3`).
  * For multivariate case, Liebnitz' rule should be prefered.
"""
import scipy as sp
import numpy as np
from eastereig.utils import faaDiBruno, diffprodTree


# Linear dependancy in lda
def Lda(k, n, dlda):
    """Compute the k-th derivative of lda(nu) with respect to nu.

    For multivariate case, the index `k` and `n` become tuple.

    **If k==n, the terms containing lda^(n) are skipped**. They dont belong
    to the RHS computation

    Parameters
    ----------
    k : int or tuple
        The requested derivative order of the flda term
    n : int or tuple
        The final requested number of derivative of lda
    dlda : iterable
        The value of the eigenvalue derivative

    """
    # k=0 nothing to du just return dlda[0]
    # to tackle mv modify with sum(k)
    if np.sum(k) == 0:
        # Use asarray to avoid trouble between list, scalar, array
        # TODO see if it the best way and uniformize the process
        return np.asarray(dlda).flat[0]
    # by convention this terms is skipped (do not belong to RHS)
    if k == n:
        return 0
    # the other are computed, and depend of the function
    else:
        return dlda[k]


def dLda(lda):
    """Compute the 1st derivative of the function Lda(nu) with respect to lda."""
    return 1.


# Quadartic dependancy in lda
def Lda2(k, n, dlda):
    """Compute the k-th derivative of  (lda(nu)**2) ie (lda(nu)**2)**(k) with respect to nu.

    For multivariate case, the index `k` and `n` become tuple. In univaraite case,
    the implementation is based on symmetric Liebnitz-rule, see arxiv.org/abs/1909.11579 Eq. 38.

    **If k==n, the terms containing lda^(n) are skipped**. They dont belong to the RHS computation

    Parameters
    ----------
    k : int or tuple
        the requested derivative order of the flda term
    n : int or tuple
        the final requested number of derivative of lda
    dlda : iterable
        the value of the eigenvalue derivative

    Examples
    --------
    Compute the derivatives of y(x)^2, with y = log(x) + 2 for x=2
    #                     0                   1                 2                   3                 4                   5
    >>> valid = np.array([7.253041736158007, 2.69314718055995, -0.846573590279973, 0.596573590279973, -0.644860385419959, 0.914720770839918])
    >>> dlda  = np.array([2.69314718055995, 0.50000000000000, -0.250000000000000, 0.250000000000000, -0.375000000000000, 0.750000000000000])
    >>> abs(Lda2(4, 5, dlda) - valid[4]) < 1e-12
    True

    Test for multivariate formalism in Univariate case
    >>> abs(Lda2((4,), (5,), dlda)- valid[4]) < 1e-12
    True

    Test for multivariate formalism in Multivariate case
    Compute the derivatives of y(x1, x2)^2, with y(x1, x2) = x1*log(x2) + 2, for x1, x2 = 0.5, 2.
    >>> dlda = np.array([[ 2.346573590279973,  0.25             , -0.125            ,  0.125            ],\
                         [ 0.693147180559945,  0.5              , -0.25             ,  0.25             ],\
                         [ 0.               ,  0.               ,  0.               ,  0.               ]])
    >>> valid = np.array([[ 5.506407614599441,  1.173286795139986, -0.461643397569993,  0.399143397569993],\
                          [ 3.253041736157983,  2.693147180559945, -0.846573590279973,  0.596573590279973],\
                          [ 0.960906027836403,  1.386294361119891,  0.306852819440055, -0.806852819440055]])
    >>> abs(Lda2((2, 1), (3, 2), dlda) - valid[2, 1]) < 1e-12
    True

    Check that for `k==n`, `Lda2((1, 3), (1, 3), dlda)` is equal to `valid[1, 3] - d lda**2/d lda * lda**(2)`
    >>> abs(Lda2((1, 3), (1, 3), dlda) - valid[1, 3] + 2*dlda.flat[0]*dlda[1, 3] ) < 1e-12
    True
    """
    # Check if multivariate
    if hasattr(n, '__iter__') or hasattr(k, '__iter__'):
        # Multivariate case
        # k=0 nothing to du just return dlda[0]**2
        if all(ki == 0 for ki in k):
            d = dlda.flat[0]**2
        else:
            if k == n:
                # Crop and put a 0 in dlda[k] to remove the term containing lda**(n)
                # which is not in the RHS
                crop = tuple(slice(0, ki+1) for ki in k)
                dlda_ = dlda[crop].copy()
                dlda_[k] = 0.
            else:
                dlda_ = dlda
            # diffprodTree compute all derivative up do the k-th, return last one
            d = diffprodTree([dlda_, dlda_], k).flat[-1]
    else:
        # Univariate case
        # This implementation is faster and clearer for univariate case
        binom = sp.special.binom
        start = 0
        # k=0 nothing to du just return dlda[0]**2
        if k == 0:
            d = dlda[0]**2
        # else compute ;-)
        else:
            if k == n: start = 1
            # init sum
            d = 0
            # upper bound
            stop = int(np.floor(k/2.))
            for j in range(start, stop+1):
                # delta is equal to one only when $n$ is even
                delta = (k/2.) == j
                d = d + binom(k, j)*(2 - delta)*dlda[k-j]*dlda[j]

    return d


def dLda2(lda):
    """Compute the 1st derivative of the function Lda2 with respect to lda."""
    return 2*lda


# Cubic dependancy in lda
def Lda3(k, n, dlda):
    r"""Compute the k-th derivative of  (lda(nu)**3) ie (lda(nu)**3)**(k) with respect to nu.

    Using Faa Di Bruno method for composite function.

    This approach is limited to scalar parameter nu.

    **If k==n, the terms containing lda^(n) are skipped**. They dont belong to the RHS computation

    Parameters
    ----------
    k : int
        The requested derivative order of the flda term.
    n : int
        The final requested number of derivative of lda.
    dlda : iterable
        The value of the eigenvalue derivative.


    Examples
    --------
    Compute the derivatives of y(x)^3, with \( y = log(x) + 2\) for x=2
    #                     0                 1                  2                 3                  4                   5
    >>> valid = np.array([19.5335089022175, 10.8795626042370, -1.40006053127857, 0.130200145858610, 0.699560166632044, -2.36641091139403 ])
    >>> dlda  = np.array([2.69314718055995, 0.50000000000000, -0.250000000000000, 0.250000000000000, -0.375000000000000, 0.750000000000000])
    >>> abs(Lda3(4, 5, dlda) - valid[4]) < 1e-12
    True

    Check that for `k==n`, `Lda3(4, 4, dlda)` is equal to `valid[4] - d lda**3/d lda * lda**(4)`
    >>> abs(Lda3(4, 4, dlda) - valid[4] + 3*dlda[0]**2*dlda[4] ) < 1e-12
    True
    """
    # Check that input argument are not iterable (not supported by faa di Bruno)
    if hasattr(n, '__iter__') or hasattr(k, '__iter__'):
        raise ValueError('Input argument cannot be an iterable.'
                         ' This implementation of `Lda3`'
                         ' cannot handle multivariate case.')

    # k=0 nothing to du just return dlda[0]**3
    if k == 0:
        d = dlda[0]**3
    # else compute ;-)
    else:
        # Create the 'outer' function derivatives
        df = np.array([dlda[0]**3, 3.*dlda[0]**2, 6.*dlda[0], 6.], dtype=dlda.dtype)
        # Append a 0 in dlda[n] to remove the term containing lda**(n) which is not known
        # thus not in the RHS
        if k == n:
            dlda_ = np.zeros(k + 1, dtype=dlda.dtype)
            dlda_[:k] = dlda[:k]
        else:
            dlda_ = dlda[:k+1]
        # Remarks : [:k+1] works even if its bigger than the vector size, it takes all
        d = faaDiBruno(df[:k+1], dlda_, k)[k]

    return d


def dLda3(lda):
    """Compute the 1st derivative of the function Lda3 with respect to lda."""
    return 3.*lda*lda


# exp(-tau * lda)
def ExpLda(k, n, dlda, tau=0.001j):
    r"""Compute the k-th derivative of  `exp(-tau*lda(nu))` with respect to nu.

    using Faa Di Bruno method for composite function.

    This approach is limited to scalar parameter nu.

    **If k==n, the terms containing lda^(n) are skipped**. They dont belong to the RHS computation

    Parameters
    ----------
    k : int
        The requested derivative order of the flda term.
    n : int
        The final requested number of derivative of lda.
    dlda : iterable
        The value of the eigenvalue derivative.
    tau : complex
        A scaling factor.

    Examples
    --------
    Compute the derivatives of exp(-tau*lda(nu)), with \( lda(nu) = nu + nu**2 + 2\) for x=2 and tau = 0.001
    #                     0                   1                    2                     3                    4
    >>> valid = np.array([0.992031914837061, -0.00496015957418530, -0.00195926303180320, 2.96369534557572e-5, 1.16073934235404e-5])
    >>> dlda  = np.array([8, 5, 2, 0, 0, 0])
    >>> abs(ExpLda(4, 5, dlda, tau=0.001) - valid[4]) < 1e-12
    True

    Check that for `k==n`, `ExpLda(4, 4, dlda)` the last terms has been removed
    >>> abs(ExpLda(2, 2, dlda, tau=0.001) - valid[2] - 0.001*ExpLda(0, 2, dlda, tau=0.001)*dlda[2] ) < 1e-12
    True
    """
    # Check that input argument are not iterable (not supported by faa di Bruno)
    if hasattr(n, '__iter__') or hasattr(k, '__iter__'):
        raise ValueError('Input argument cannot be an iterable.'
                         ' This implementation of `Lda3`'
                         ' cannot handle multivariate case.')
    # k=0 nothing to do
    if k == 0:
        d = np.exp(-tau*dlda[0])
    # else compute ;-)
    else:
        # Create the 'outer' function derivatives
        f = np.exp(-tau*dlda[0])
        df = (-tau)**np.arange(0, len(dlda))*f
        # Append a 0 in dlda[n] to remove the term containing lda**(n) which is not known
        # thus not in the RHS
        if k == n:
            dlda_ = np.zeros(k + 1, dtype=complex)
            dlda_[:k] = dlda[:k]
        else:
            dlda_ = dlda[:k+1]
        # Remarks : [:k+1] works even if its bigger than the vector size, it takes all
        d = faaDiBruno(df[:k+1], dlda_, k)[k]

    return d


def dExpLda(lda):
    """Compute the 1st derivative of the function exp(-tau*lda) with respect to lda."""
    # FIXME find a better way to do that...
    # trick to recover the defaut value of tau in ExpLda
    tau = ExpLda.__defaults__[0]
    return -tau*np.exp(-tau*lda)


# Mapping between f(lda) -> d_\dlda f(lda)
_dlda_flda = {Lda: dLda,
              Lda2: dLda2,
              Lda3: dLda3,
              ExpLda: dExpLda,
              None: lambda x: 0,
              }
r"""
Define a dict to map the derivative with respect to lda \( \partial_\lambda f(\lambda) \) and
the function of lda \( f(\lambda) \) .
If new function is added below, a link to its derivative must be be added here.
"""

# %% Main for basic tests
if __name__ == '__main__':
    # run doctest Examples
    import doctest
    doctest.testmod()
