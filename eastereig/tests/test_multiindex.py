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

""" Test with sympy the computation of ultinomial_multiindex_coefficients.
"""

import unittest
import numpy as np
import eastereig.utils as utils
try:
    import sympy as sym
except ImportError:
    print('Sympy is required to run this test.')


class Test_multinomial_multiindex_coefficients(unittest.TestCase):
    """ Define calculated question parser test cases for unittest.
    """
    @classmethod
    def setUpClass(cls):
        print('\n> Tests of ', cls.__name__)

    def setUp(self):

        sym.init_printing(forecolor='White')

    def test_product_of_2_func(self):
        """ Test multinomial_multiindex_coefficients with a product of 2 functions.
        """

        N = (2, 3)
        x, y = sym.symbols('x, y')
        f, g = sym.Function('f')(x, y), sym.Function('g')(x, y)

        # 2 functions f*g
        m = 2
        mmi_index, mmi_coef = utils.multinomial_multiindex_coefficients(m, N)
        # compute reference solution with sympy
        d = sym.diff(f*g, x, N[0], y, N[1])

        # built sympy expression from ee coefs and index
        d_ = 0
        for i, c in zip(mmi_index, mmi_coef):
            d_ += c * sym.diff(f, x, i[0][0], y, i[0][1]) \
                    * sym.diff(g, x, i[1][0], y, i[1][1])
        # check
        self.assertTrue(d_ == d)

    def test_product_of_3_func(self):
        """ Test multinomial_multiindex_coefficients with a product of 3 functions.
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


if __name__ == '__main__':
    # run unittest test suite
    unittest.main()
