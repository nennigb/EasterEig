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

"""Unittest test suite for charpol module.
"""

import unittest
import numpy as np
import eastereig as ee
from eastereig.examples.WGadmitance_numpy_mv import Ynumpy


class Test_charpol_mult(unittest.TestCase):
    """ Define multinomial multiindex coefficients test cases.
    """
    @classmethod
    def setUpClass(cls):
        print('\n> Tests of ', cls.__name__)

    def setUp(self):
        # Take a small number of nodes for rapidity
        N = 20
        # Set all constant to 1 for simplicity
        rho0, c0, k0, h = 1., 1., 1., 1.
        # Initial admittance
        nu0_ref = np.array([3.1781 + 4.6751j, 3.0875 + 3.6234j]) / 1j
        nu0 = np.array([1.5, 0.8])*nu0_ref
        # number of mode to compute
        Nmodes = 6

        # Create discrete operator of the pb
        imp = Ynumpy(y=nu0, n=N, h=h, rho=rho0, c=c0, k=k0)
        # Initialize eigenvalue solver for *generalized eigenvalue problem*
        imp.createSolver(pb_type='gen')
        # run the eigenvalue computation
        Lambda = imp.solver.solve(nev=Nmodes, target=0+0j, skipsym=False)
        # create a list of the eigenvalue to monitor
        lda_list = np.arange(0, Nmodes)
        # return the eigenvalue and eigenvector in a list of Eig object
        extracted = imp.solver.extract(lda_list)
        # destroy solver (important for petsc/slepc)
        imp.solver.destroy()
        # Number of derivative
        Nderiv = (4, 4)
        # Get eigenvalues derivatives
        for vp in extracted:
            vp.getDerivativesMV(Nderiv, imp)
        # Store them
        self.extracted = extracted
        self.imp = imp

    def test_charpol_mult_same_direct_computation(self):
        """ Test that `multiply` yield same results that direct computation.
        """
        extracted = self.extracted

        # Create globals and partial CharPol
        C05 = ee.CharPol(extracted[0:6])
        C04 = ee.CharPol(extracted[0:5])

        C02 = ee.CharPol(extracted[0:3])
        C35 = ee.CharPol(extracted[3:6])
        C34 = ee.CharPol(extracted[3:5])
        # Check with 2 polynomials of the same size
        P = C02.multiply(C35)
        print('\n')
        # Check size
        self.assertTrue(len(C05.dcoefs) == len(P.dcoefs))

        # Check values
        check_an = np.zeros((len(C05.dcoefs),), dtype=bool)
        for i, (an, bn) in enumerate(zip(C05.dcoefs, P.dcoefs)):
            check_an[i] = np.allclose(an, bn)
        self.assertTrue(check_an.all())

        # Check operator form for asymetric polynomials
        Q = C02 * C34
        # Check values
        check_an = np.zeros((len(C04.dcoefs),), dtype=bool)
        for i, (an, bn) in enumerate(zip(C04.dcoefs, Q.dcoefs)):
            check_an[i] = np.allclose(an, bn)
        self.assertTrue(check_an.all())

    def test_factory_from_recursive_mult(self):
        """ Test that `_from_recursive_mult` yield same CharPol than Vieta.
        """
        extracted = self.extracted

        # Create globals and partial CharPol
        Cv = ee.CharPol(extracted[0:6])
        Cr2 = ee.CharPol._from_recursive_mult(extracted[0:6], block_size=2)
        Cr3 = ee.CharPol._from_recursive_mult(extracted[0:6], block_size=3)

        # Check all polynomials have the same size
        self.assertTrue(len(Cv.dcoefs) == len(Cr2.dcoefs))
        self.assertTrue(len(Cv.dcoefs) == len(Cr3.dcoefs))

        # Check values
        check_r2 = np.zeros((len(Cv.dcoefs),), dtype=bool)
        check_r3 = np.zeros((len(Cv.dcoefs),), dtype=bool)
        for i, (an, bn, cn) in enumerate(zip(Cv.dcoefs, Cr2.dcoefs, Cr3.dcoefs)):
            check_r2[i] = np.allclose(an, bn)
            check_r3[i] = np.allclose(an, cn)
        self.assertTrue(check_r2.all())
        self.assertTrue(check_r3.all())

if __name__ == '__main__':
    # run unittest test suite
    unittest.main()
