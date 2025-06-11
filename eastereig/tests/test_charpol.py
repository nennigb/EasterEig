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
import time
import tempfile
from os import path
import sympy as sym
from eastereig.examples.WGadmitance_numpy_mv import Ynumpy, nu_ref
from eastereig.examples.WGimpedance_numpy import Znumpy, z_ref
from eastereig.examples.toy_3dof_2params import Network, Jana, vana, sympy_check_poly
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from importlib.util import find_spec as _find_spec
from tempfile import TemporaryDirectory

class Test_charpol_mult(unittest.TestCase):
    """Define multivariate charpol test cases.
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
        """Test that `multiply` yields the same results that the direct computation.

        It tests the behavior in parallel and in sequentiel.
        """
        extracted = self.extracted

        # Create globals and partial CharPol
        C05 = ee.CharPol(extracted[0:6])
        C04 = ee.CharPol(extracted[0:5])

        C02 = ee.CharPol(extracted[0:3])
        C35 = ee.CharPol(extracted[3:6])
        C34 = ee.CharPol(extracted[3:5])

        # Check with 2 polynomials of the same size in sequential ------------
        P = C02.multiply(C35, max_workers=1)
        print('\n')
        # Check size
        self.assertTrue(len(C05.dcoefs) == len(P.dcoefs))

        # Check values
        check_an = np.zeros((len(C05.dcoefs),), dtype=bool)
        for i, (an, bn) in enumerate(zip(C05.dcoefs, P.dcoefs)):
            check_an[i] = np.allclose(an, bn)
        self.assertTrue(check_an.all())

        # Check with 2 polynomials of the same size in parallel --------------
        P = C02.multiply(C35, max_workers=4)
        print('\n')
        # Check size
        self.assertTrue(len(C05.dcoefs) == len(P.dcoefs))

        # Check values
        check_an = np.zeros((len(C05.dcoefs),), dtype=bool)
        for i, (an, bn) in enumerate(zip(C05.dcoefs, P.dcoefs)):
            check_an[i] = np.allclose(an, bn)
        self.assertTrue(check_an.all())

        # Check operator form for asymetric polynomials ----------------------
        # Deactivate parallelism in the charpol `multiply` method
        ee.options.gopts['max_workers_mult'] = 1
        Q = C02 * C34
        # Check values
        check_an = np.zeros((len(C04.dcoefs),), dtype=bool)
        for i, (an, bn) in enumerate(zip(C04.dcoefs, Q.dcoefs)):
            check_an[i] = np.allclose(an, bn)
        self.assertTrue(check_an.all())
        # Activate parallelism in the charpol `multiply` method
        ee.options.gopts['max_workers_mult'] = 2
        Q = C02 * C34
        # Check values
        check_an = np.zeros((len(C04.dcoefs),), dtype=bool)
        for i, (an, bn) in enumerate(zip(C04.dcoefs, Q.dcoefs)):
            check_an[i] = np.allclose(an, bn)
        self.assertTrue(check_an.all())

    def test_factory_from_recursive_mult(self):
        """Test that `from_recursive_mult` yields the same CharPol than Vieta.
        """
        extracted = self.extracted
        # Create globals and partial CharPol
        Cv = ee.CharPol(extracted[0:6])
        t0 = time.time()
        Cr2 = ee.CharPol.from_recursive_mult(extracted[0:6], block_size=2)
        Cr3 = ee.CharPol.from_recursive_mult(extracted[0:6], block_size=3)
        print('time =', time.time() - t0)

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

    def test_charpol_set_param(self):
        """Test the `set_param` method.
        """
        extracted = self.extracted
        # Create globals and partial CharPol
        C = ee.CharPol.from_recursive_mult(extracted[0:4])
        nu0 = C.nu0
        C1 = C.set_param(1, nu0[1] + 0.1)
        C0 = C.set_param(0, nu0[0] - 0.1)
        ref = C.eval_an_at(np.array(nu0) + np.array([-0.1, 0.1]))
        self.assertTrue(np.allclose(C0.eval_an_at(nu0[1] + 0.1), ref))

    def test_load_and_export(self):
        """Test the `load` and `export` method.
        """
        extracted = self.extracted
        filename = 'PCP.npz'
        # Create globals and partial CharPol
        C = ee.CharPol.from_recursive_mult(extracted[0:4])
        C_lda = C.eval_lda_at(C.nu0 + 0.1)
        with tempfile.TemporaryDirectory() as tmpdirname:
            C.export(path.join(tmpdirname, filename))
            C2 = ee.CharPol.load(path.join(tmpdirname, filename))
            C2_lda = C2.eval_lda_at(C2.nu0 + 0.1)
        C_lda.sort()
        C2_lda.sort()
        self.assertTrue(np.allclose(C_lda, C2_lda))

    @unittest.skipUnless(_find_spec('pypolsys'), 'pypolsys is an optional solver')
    def test_homotopy_solver(self):
        """Test the homotopy (optional) EP solver.
        """
        C = ee.CharPol.from_recursive_mult(self.extracted[0:5])
        bplp, s = C.homotopy_solve(tracktol=1e-12, finaltol=1e-13, tol_filter=-1)
        delta, sol, deltaf = C.filter_spurious_solution(s, plot=False, tol=1e-1)
        # Compute error with reference solution
        D = distance_matrix(sol[:, 1][:, None], nu_ref[:, None])
        row_ind, col_ind = linear_sum_assignment(D)
        error = D[row_ind, col_ind].sum()
        self.assertTrue(error < 5e-2)


class Test_charpol_uni(unittest.TestCase):
    """Define univariate charpol test cases.
    """
    @classmethod
    def setUpClass(cls):
        print('\n> Tests of ', cls.__name__)

    def setUp(self):
        # Number of derivative
        Nderiv = 12
        # air properties
        rho0, c0 = 1.2, 340.
        # initial imepdance guess
        z0 = 486.198103097114 + 397.605679264872j
        # freq. of computation
        f = 200
        # number of dof
        N = 50
        # duct height
        h = 1.
        # number of mode to compute
        Nmodes = 5
        # solve the problem
        omega = 2*np.pi*f
        k0 = omega/c0
        # Create discrete operator of the pb
        imp = Znumpy(z=z0, n=N, h=h, rho=rho0, c=c0, k=k0)
        print(imp)
        # initialize eigenvalue solver for *generalized eigenvalue problem*
        imp.createSolver(pb_type='gen')
        # run the eigenvalue computation
        Lambda = imp.solver.solve(nev=Nmodes, target=0+0j, skipsym=False)
        # return the eigenvalue and eigenvector in a list of Eig object
        extracted = imp.solver.extract(range(0, Nmodes+1))
        # destroy solver (important for petsc/slepc)
        imp.solver.destroy()
        # Get eigenvalues derivatives
        for vp in extracted:
            vp.getDerivatives(Nderiv, imp)

        # Store them
        self.extracted = extracted
        self.imp = imp

    def test_charpol_uni_with_ref(self):
        """Test that Charpol can find EP for impedance test case.
        """
        # Locate EP using Charpol
        C = ee.CharPol.from_recursive_mult(self.extracted)
        bplp, sol_h = C.homotopy_solve(tracktol=1e-14, finaltol=1e-8, tol_filter=-1)
        delta, sol, delta_sol = C.filter_spurious_solution(sol_h, plot=False, tol=1)
        # Check if charpol find the EP
        self.assertTrue(np.allclose(sol[:, 1], np.array(z_ref[:sol.shape[0]]), rtol=1e-2, atol=-1))

    def test_dh_uni_with_ref(self):
        """Test that dh can find EP for impedance test case.
        """
        # Locate EP using Charpol
        C = ee.CharPol.from_recursive_mult(self.extracted)
        dh = C.getdH()
        r, s = dh.locate()
        self.assertTrue(np.allclose(r[0], np.array(z_ref[0])))

class Test_charpol_basics(unittest.TestCase):
    """Define charpol basic function test cases.
    """
    @classmethod
    def setUpClass(cls):
        print('\n> Tests of ', cls.__name__)

    def setUp(self):
        # Define the order of derivation for each parameter
        Nderiv = (5, 5)
        # Define the inital value of parameter
        nu0 = np.array([0.95590969 - 1.48135044j + 0.15 + 0.1j,
                        1. + 1.41421356e+00j + 0.1 + 0.1j])
        # Instiate the problem
        net = Network(nu0)
        net.createSolver(pb_type='gen')
        # Run the eigenvalue computation
        Lambda = net.solver.solve(target=0+0j)
        # Create a list of the eigenvalue to monitor
        lda_list = np.arange(0, 3)
        # Return the eigenvalue and eigenvector in a list of Eig object
        extracted = net.solver.extract(lda_list)
        # destroy solver (important for petsc/slepc)
        net.solver.destroy()
        # Compute the eigenvalue derivatives
        for vp in extracted:
            vp.getDerivativesMV(Nderiv, net)
        # Find EP with Charpol
        C = ee.CharPol(extracted)
        # Store them
        self.C = C
        self.s = (1.5, 1., 2.)

    def test_charpol_jacobian(self):
        """Test jacobian matrix wtr to sympy.
        """
        # Locate EP using Charpol
        Jpcp = self.C.jacobian(self.s)
        # Check if charpol find the EP
        self.assertTrue(np.allclose(Jpcp, Jana, rtol=1e-6))

    def test_charpol_discrimant(self):
        """Test discriminant computation wtr to sympy.
        """
        s = self.s
        # Use Sylvester matrix
        d_pcp_syl = self.C.discriminant(s[1::])
        p0, mu, nu, lda = sympy_check_poly()
        p = sym.Poly(p0, lda)
        d_ana = p.discriminant().subs({mu: s[1], nu: s[2]})
        # Use Taylor expansion
        H = self.C.getdH()
        d_dH = H.eval_at(s[1::])
        # Check results
        self.assertTrue(abs(d_ana-d_pcp_syl) < 1e-5)
        self.assertTrue(abs(d_ana-d_dH) < 1e-5)

    def test_charpol_load_export(self):
        """Test discriminant export and load.
        """
        temp_dir = TemporaryDirectory()
        self.C.export(path.join(temp_dir.name, 'pcp'))
        D = ee.CharPol.load(path.join(temp_dir.name, 'pcp'))
        self.assertTrue(np.allclose(self.C.EP_system(self.s),
                                    D.EP_system(self.s), rtol=1e-6))
        temp_dir.cleanup()

if __name__ == '__main__':
    # run unittest test suite
    unittest.main()
