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
is provide.

All evaluations are performed for *complex* argument nu.
In all cases the coefficients given in the array a
the variable nu[0] is associated the `a` 1st dimension, nu[1] the second,
and nu[2] the third.
"""

# from . import fpolyval as fp
from eastereig.fpoly import fpolyval
import numpy as np
from numpy.polynomial.polynomial import polyval, polyval2d
import numpy.polynomial.polyutils as pu

def polyvalnd(nu, a):
    """ Evaluate a multivariate polynomial at complex vector nu, using horner
    method.

    The convention a_ij nu**i * y**j, where a_00 is the constant and a[-1,-1]
    is the highest degree term.

    The ordering is the same as numpy.polynomial.polynomial polyval*
    These module is used if #nu > 3.

    Parameters
    ----------
    nu: iterable.
        Complex value where the polynomial should be evaluated.
    a: ndarray
        Complex value array with the polynomial coefficient.

    Returns
    -------
    pval : Complex
        The polynomial value.

    Examples
    --------
    >>> a2 = np.random.rand(5, 5)*(1+1j)
    >>> pvalf = polyvalnd((2.253+0j, 1+1j), a2)
    >>> pvalnp = polyval2d(2.253+0j, 1+1j, a2)
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
    else:
        pval = pu._valnd(polyval, a, *nu)

    return pval


# %% Main for basic tests
if __name__ == '__main__':
    # run doctest Examples
    import doctest
    doctest.testmod()