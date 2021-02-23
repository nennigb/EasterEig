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

""" Define the functions to compute the successive derivatives of the function of the eigenvalue
with respect to nu. These functions appear in
L=K_0 * None + K_1*Lda(nu) + K_2 * Lda2(nu) + ... + K_n * f_n(lda(nu))

  - These functions sould have the same interface : my_fun(k, n, dlda)
  - **If k==n, the terms containing lda^(n) are skipped**. They dont belong to the RHS computation
  - a companion function providing the derivitive with respect to lda should be also defined and
  enrolled in `dlda_flda` dict (at the end) define a dict to get the derivative % lda of the
  function of lda

Remarks:
--------
1. Only linear and quadratic dependancy are now implemented
2. If function are added, don't forget do update `_dlda_flda`

"""
import scipy as sp
import numpy as np
from eastereig.utils import faaDiBruno

# linear dependancy in lda
def Lda(k, n, dlda):
    """
    Compute the k-th derivative of lda(nu) with respect to nu.

    **If k==n, the terms containing lda^(n) are skipped**. They dont belong
    to the RHS computation

    Parameters
    -----------
    k : int
        the requested derivative order of the flda term
    n : int
        the final requested number of derivative of lda
    dlda : iterable
        the value of the eigenvalue derivative

    """
    # k=0 nothing to du just return dlda[0]
    # to tackle mv modify with sum(k)
    if np.sum(k) == 0:
        return dlda[0]
    # by convention this terms is skipped (do not belong to RHS)
    if k == n:
        return 0
    # the other are computed, and depend of the function
    else:
        return dlda[k]


def dLda(lda):
    """ Compute the 1st derivative of the function Lda(nu) with respect to lda.
    """
    return 1.


# quadartic dependancy in lda
def Lda2(k, n, dlda):
    """ Compute the k-th derivative of  (lda(nu)**2) ie (lda(nu)**2)**(k) with respect to nu
    based on liebnitz-rule, see arxiv.org/abs/1909.11579 Eq. 38.

    **If k==n, the terms containing lda^(n) are skipped**. They dont belong to the RHS computation

    Parameters
    -----------
    k : int
        the requested derivative order of the flda term
    n : int
        the final requested number of derivative of lda
    dlda : iterable
        the value of the eigenvalue derivative

    Examples
    ---------
    Compute the derivatives of y(x)^2, with \( y = log(x) + 2\) for x=2
    #                     0                   1                 2                   3                 4                   5
    >>> valid = np.array([7.253041736158007, 2.69314718055995, -0.846573590279973, 0.596573590279973, -0.644860385419959, 0.914720770839918])
    >>> dlda  = np.array([2.69314718055995, 0.50000000000000, -0.250000000000000, 0.250000000000000, -0.375000000000000, 0.750000000000000])
    >>> abs(Lda2(4, 5, dlda) - valid[4]) < 1e-12
    True
    """
    binom = sp.special.binom
    start = 0

    # k=0 nothing to du just return dlda[0]**2
    if k == 0:
        d = dlda[0]**2
    # else compute ;-)
    else:
        if k == n: start=1
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
    """ Compute the 1st derivative of the function Lda2 with respect to lda.
    """
    return 2*lda


# Cubic dependancy in lda
def Lda3(k, n, dlda):
    """ Compute the k-th derivative of  (lda(nu)**3) ie (lda(nu)**3)**(k) with respect to nu
    using Faa Di Bruno method for composite function.

    **If k==n, the terms containing lda^(n) are skipped**. They dont belong to the RHS computation

    Parameters
    -----------
    k : int
        The requested derivative order of the flda term.
    n : int
        The final requested number of derivative of lda.
    dlda : iterable
        The value of the eigenvalue derivative.


    Examples
    ---------
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
    """ Compute the 1st derivative of the function Lda3 with respect to lda.
    """
    return 3.*lda*lda


# Quartic dependancy in lda
def Lda4(k, n, dlda):
    """ Compute the k-th derivative of  (lda(nu)**4) ie (lda(nu)**4)**(k) with respect to nu
    using Faa Di Bruno method for composite function.

    **If k==n, the terms containing lda^(n) are skipped**. They dont belong to the RHS computation

    Parameters
    -----------
    k : int
        The requested derivative order of the flda term.
    n : int
        The final requested number of derivative of lda.
    dlda : iterable
        The value of the eigenvalue derivative.

    Examples
    ---------
    Compute the derivatives of y(x)^4, with \( y = log(x) + 2\) for x=2
    #                     0                 1                  2                 3                  4                   5
    >>> valid = np.array([52.6066144264496, 39.0670178044350, 2.22561630625647, -5.02573736881360, 7.79900634493763, -14.1988923566112])
    >>> dlda  = np.array([2.69314718055995, 0.50000000000000, -0.250000000000000, 0.250000000000000, -0.375000000000000, 0.750000000000000])
    >>> abs(Lda4(4, 5, dlda) - valid[4]) < 1e-12
    True

    Check that for `k==n`, `Lda4(4, 4, dlda)` is equal to `valid[4] - d lda**4/d lda * lda**(4)`
    >>> abs(Lda4(4, 4, dlda) - valid[4] + 4*dlda[0]**3*dlda[4]) < 1e-12
    True
    """
    # k=0 nothing to du just return dlda[0]**3
    if k == 0:
        d = dlda[0]**4
    # else compute ;-)
    else:
        # Create the 'outer' function derivatives
        df = np.array([dlda[0]**4, 4*dlda[0]**3, 12.*dlda[0]**2, 24.*dlda[0], 24.],
                      dtype=dlda.dtype)
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


def dLda4(lda):
    """ Compute the 1st derivative of the function Lda4 with respect to lda.
    """
    return 4.*lda**3


# Mapping between f(lda) -> d_\dlda f(lda)
_dlda_flda = {Lda: dLda,
              Lda2: dLda2,
              Lda3: dLda3,
              Lda4: dLda4,
              None: lambda x: 0,
              }
r"""
Define a dict to map the derivative with respect to lda \( \partial_\lambda f(\lambda) \) and
the function of lda \( f(\lambda) \) .
If new function is added below, a link to its derivative must be be added here.
`faaDiBruno` method allows to compute the derivative for a large variety of composite
function \( f(\lambda (\nu)) \).
"""
