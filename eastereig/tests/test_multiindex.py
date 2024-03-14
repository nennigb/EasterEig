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

"""Test with sympy the computation of multinomial_multiindex_coefficients.
"""

import unittest
import numpy as np
import eastereig.utils as utils
try:
    import sympy as sym
except ImportError:
    print('Sympy is required to run this test.')


class Test_multinomial_multiindex_coefficients(unittest.TestCase):
    """Define multinomial multiindex coefficients test cases.
    """
    @classmethod
    def setUpClass(cls):
        print('\n> Tests of ', cls.__name__)

    def setUp(self):
        sym.init_printing(forecolor='White')

    def test_product_of_2_func(self):
        """Test multinomial_multiindex_coefficients with a product of 2 functions.
        """
        N = (2, 3)
        x, y = sym.symbols('x, y')
        f, g = sym.Function('f')(x, y), sym.Function('g')(x, y)

        # 2 functions f*g
        m = 2
        mmi_index, mmi_coef = utils.multinomial_multiindex_coefficients(m, N)
        # compute reference solution with sympy
        d = sym.diff(f*g, x, N[0], y, N[1])

        # built sympy expressions from ee coefs and index
        d_ = 0
        for i, c in zip(mmi_index, mmi_coef):
            d_ += c * sym.diff(f, x, i[0][0], y, i[0][1]) \
                    * sym.diff(g, x, i[1][0], y, i[1][1])
        # check
        self.assertTrue(d_ == d)

    def test_product_of_3_func(self):
        """Test multinomial_multiindex_coefficients with a product of 3 functions.
        """
        N = (2, 3)
        x, y = sym.symbols('x, y')
        f, g, h = sym.Function('f')(x, y), sym.Function('g')(x, y), sym.Function('h')(x, y)

        # 3 functions f*g
        m = 3
        mmi_index, mmi_coef = utils.multinomial_multiindex_coefficients(m, N)
        # compute reference solution with sympy
        d = sym.diff(f*g*h, x, N[0], y, N[1])

        # built sympy expression from ee coefs and index
        d_ = 0
        for i, c in zip(mmi_index, mmi_coef):
            d_ += c * sym.diff(f, x, i[0][0], y, i[0][1]) \
                    * sym.diff(g, x, i[1][0], y, i[1][1]) \
                    * sym.diff(h, x, i[2][0], y, i[2][1])
        # check
        self.assertTrue(d_ == d)

    def test_diffprodMV_and_diffprodTreeMV(self):
        """Test the computation derivative of multivariate product of functions.

        example : H = (x*y**2) * exp(x*y) @ x=0.5, y=1.5
        """
        tol = 1e-10
        # derivation order (Cannot be changed here)
        N = (2, 3)
        x = 0.5
        y = 1.5

        dh0 = np.array([[x*y**2, x*2*y, x*2, 0, 0],
                        [y**2,   2*y,   2,   0, 0],
                        [0,      0,     0,   0, 0],
                        [0,      0,     0,   0, 0]])
        exy = np.exp(x*y)
        xy = x*y

        dh1 = np.array([[exy,   x * exy, x**2 * exy, x**3 * exy, x**4*exy],
                        [y*exy, (xy + 1)*exy,  x*(xy + 2)*exy, x**2 * (xy + 3)*exy, x**3*(xy + 4)*exy],
                        [y**2 * exy, y*(xy + 2)*exy,   (x**2 * y**2 + 4*xy + 2) * exy, x*(x**2 * y**2 + 6*xy + 6) * exy, x**2*(x**2*y**2 + 8*xy + 12)*exy],
                        [y**3*exy, y**2*(xy + 3)*exy, y*(x**2*y**2 + 6*xy + 6)*exy, (x**3*y**3 + 9*x**2*y**2 + 18*xy + 6)*exy, x*(x**3*y**3 + 12*x**2*y**2 + 36*xy + 24)*exy]])
        # 'numerical' derivation with ee Liebnitz (standard)
        dH = utils.diffprodMV([dh0, dh1], N)
        # 'numerical' derivation with ee Liebnitz (optimized)
        dHt = utils.diffprodTreeMV([dh0, dh1], N)

        # Compute sympy ref solution
        x_, y_ = sym.symbols('x, y')
        H_ = (x_*y_**2) * sym.exp(x_*y_)
        dH_12 = sym.diff(H_, x_, 1, y_, 2)
        dH_23 = sym.diff(H_, x_, 2, y_, 3)

        # check with sympy
        self.assertTrue(abs(dH[1, 2] - sym.N(dH_12.subs({x_: x, y_: y}))) < tol)
        self.assertTrue(abs(dH[2, 3] - sym.N(dH_23.subs({x_: x, y_: y}))) < tol)
        self.assertTrue(abs(dHt[1, 2] - sym.N(dH_12.subs({x_: x, y_: y}))) < tol)
        self.assertTrue(abs(dHt[2, 3] - sym.N(dH_23.subs({x_: x, y_: y}))) < tol)


if __name__ == '__main__':
    # run unittest test suite
    unittest.main()
