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
Create a common interface to `fpolyval` fortran function.

Due to the possibly hudge number of polynomial evaluation, fortran implementation
is provided.

All evaluations are performed for *complex* argument nu.
In all cases the coefficients given in the array a
the variable nu[0] is associated the `a` 1st dimension, nu[1] the second,
and nu[2] the third.
"""

# from . import fpolyval as fp
from eastereig.fpoly import fpolyval
import numpy as np
import numpy.polynomial.polyutils as pu
from numpy.polynomial.polynomial import polyval


def polyvalnd(nu, a):
    """Evaluate a multivariate polynomial at complex vector nu.

    This module use the series' convention, ie considering a polynomial such
    P = sum_{i,...,k} a_i..k nu[0]**i ... nu[k]**k, the term a_00 is the
    constant and a[-1,..., -1] is the highest degree term.
    The ordering is the same as numpy.polynomial.polynomial.

    It use Horner method and map the fortran functions from `fpolyval.f90`
    if #nu <= 4. For higher number of variables, `numpy.polynomial.polynomial`
    is used.


    Parameters
    ----------
    nu: iterable.
        Complex vector where the polynomial should be evaluated.
    a: ndarray
        Complex valued array with the polynomial coefficients.

    Returns
    -------
    pval : Complex
        The polynomial value.

    Examples
    --------
    Validation examples using numpy
    >>> from numpy.polynomial.polynomial import polyval
    >>> a1 = np.random.rand(5)*(1+1j)
    >>> pvalf = polyvalnd(2.253+0.1j, a1)
    >>> pvalnp = polyval(2.253+0.1j, a1)
    >>> abs(pvalf - pvalnp) < 1e-12
    True

    >>> from numpy.polynomial.polynomial import polyval2d
    >>> a2 = np.random.rand(5, 5)*(1+1j)
    >>> pvalf = polyvalnd((2.253+0j, 1+1j), a2)
    >>> pvalnp = polyval2d(2.253+0j, 1+1j, a2)
    >>> abs(pvalf - pvalnp) < 1e-12
    True

    >>> from numpy.polynomial.polynomial import polyval3d
    >>> a3 = np.random.rand(5, 5, 5)*(1+1j)
    >>> pvalf =  polyvalnd((2.253, 1+1j, 0.1+2j), a3)
    >>> pvalnp = polyval3d(2.253, 1+1j, 0.1+2j, a3)
    >>> abs(pvalf - pvalnp) < 1e-12
    True

    >>> a4 = np.random.rand(5, 5, 5, 5)*(1+1j)
    >>> pvalf =  polyvalnd((2.253, 1+1j, 0.1+2j, -0.8+0.2j), a4)
    >>> pvalnp = pu._valnd(polyval, a4, 2.253, 1+1j, 0.1+2j, -0.8+0.2j)
    >>> abs(pvalf - pvalnp) < 1e-12
    True
    """
    d = len(a.shape)
    if d == 1:
        pval = fpolyval.fpolyval(nu, a)
    elif d == 2:
        pval = fpolyval.fpolyval2(*nu, a)
    elif d == 3:
        pval = fpolyval.fpolyval3(*nu, a)
    elif d == 4:
        pval = fpolyval.fpolyval4(*nu, a)
    else:
        # Not implemented. Use numpy instead
        pval = pu._valnd(polyval, a, *nu)

    return pval


# %% Main for basic tests
if __name__ == '__main__':
    # run doctest Examples
    import doctest
    doctest.testmod()
