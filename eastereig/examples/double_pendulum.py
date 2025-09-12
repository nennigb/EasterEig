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
Coupled pendulum example.

    #####################
        /           /
       /           /
      /    k12    /
     /----www----/
    /           /
   / θ1, l1    /  θ2, l2
 ,─.         ,─.           | g
( m1)       ( m2)          v
 `-'         `-'

For a full description of the problem, see
> N. Even, B. Nennig, G. Lefebvre and E. Perrey-Debain.
> Experimental observation of exceptional points in coupled pendulums.
> Journal of Sound and Vibration, (2023).
> doi:10.1016/j.jsv.2024.118239, open access: arXiv:2308.01089.

For a finding EP2 with two real parameters, see
> https://github.com/nicolase7en/real-valued-ep2
this approach is based on multivariate polynomial and numerical continuation.

Examples
--------
>>> C, sol, err_c, err_r1, err_r_all = main() # doctest: +ELLIPSIS
> Solve PEP eigenvalue problem with NumpyEigSolver class...
>>> err_c[0] < 1e-4    # Starting point closer
True
>>> err_c[1] < 5e-2    # Starting point farther
True
>>> err_r1 < 1e-4      # global error, closest EP
True
>>> err_r_all < 1e-2   # global error, all EP
True
"""
import numpy as np
import eastereig as ee
norm = np.linalg.norm

# References solutions, obtained with https://github.com/nicolase7en/real-valued-ep2/coupled_pendulums.py
# For real parameters
l2_ref1 = 0.6748427292149384
lda_ref1 = - 0.023143351994887016 + 3.271191554228187j
c2_ref1 = 0.132987642233874

l2_ref2 = 0.24156299318770963
lda_ref2 = - 0.027040743364087093 + 3.271155054323085j
c2_ref2 = 0.11318451798963342


class Pendulum(ee.OPmv):
    """Define the coupled pendulum problem."""

    def __init__(self, nu0):
        """Initialize the problem.

        Parameters
        ----------
            nu0 : tuple
                The parameters initial Values (complex).
        """
        # Parameters values
        self.param = {
            'm1': 1,
            'l1': 0.9534116593508425,
            'moi1': 0.05627128550779616,  # interia of the rod
            'm2': 1,
            # l2 = 0.6749721473410476
            'moi2': 1.0001497454962438,  # interia of the rod and plate
            'w1': 3.7821204567028786,
            'w2': 2.9638804625382984,
            'k12': 0.1750615764438783,
            'c1': 1e-03,
            # c2 = 1e-02
            'c12': 1e-04,
            'g': 9.80665,
        }

        # initialize OP interface
        self.setnu0(nu0)

        # mandatory -----------------------------------------------------------
        self._lib = 'numpy'
        # create the operator matrices
        self.K = [self._K(), self._C(), self._M()]
        # define the list of function to compute  the derivatives of each operator matrix
        self.dK = [self._dK, self._dC, self._dM]
        # define the list of function to set the eigenvalue dependance of each operator matrix
        self.flda = [None, ee.lda_func.Lda, ee.lda_func.Lda2]
        # ---------------------------------------------------------------------

    def __repr__(self):
        """Define the object representation."""
        text = "Instance of Operator class {} @nu0={}."
        return text.format(self.__class__.__name__, self.nu0)

    def _M(self):
        """Define the mass matrix."""
        l2, c2 = self.nu0
        p = self.param
        M = np.array([[p['m1'] * p['l1']**2 + p['moi1'], 0],
                      [0, p['m2'] * l2**2 + p['moi2']]], dtype=complex)
        return M

    def _K(self):
        """Define the stiffness matrix."""
        l2, c2 = self.nu0
        p = self.param
        K = np.array([[p['m1'] * p['g'] * p['l1'] + p['w1']**2 * p['moi1'] + p['k12'], -p['k12']],
                      [-p['k12'], p['m2'] * p['g'] * l2 + p['w2']**2 * p['moi2'] + p['k12']]], dtype=complex)
        return K

    def _C(self):
        """Define the damping matrix."""
        l2, c2 = self.nu0
        p = self.param
        C = np.array([[p['c1'] + p['c12'], - p['c12']],
                      [-p['c12'], c2 + p['c12']]], dtype=complex)
        return C

    def _dK(self, m, n):
        r"""Define the sucessive derivative of the $K$ matrix with respect to nu.

        Parameters
        ----------
        m, n : int
            The order of derivation for nu[0] and nu[1].

        Returns
        -------
        Kn : Matrix (petsc or else)
            The n-derivative of global K0 matrix
        """
        mu, nu = self.nu0
        p = self.param
        if (m, n) == (0, 0):
            Kn = self.K[0]
        elif (m, n) == (1, 0):
            Kn = np.array([[0, 0],
                           [0, p['m2'] * p['g']]], dtype=complex)
        # if (m, n) > (1, 0) return 0 because K has a linear dependancy on l2
        else:
            return 0
        return Kn

    def _dM(self, m, n):
        """Define the sucessive derivative of the $M$ matrix with respect to nu.

        Parameters
        ----------
        m, n  : int
            The order of derivation for nu[0] and nu[1].

        Returns
        -------
        : Matrix (petsc or else)
            The n-derivative of global K1 matrix
        """
        l2, c2 = self.nu0
        p = self.param

        if (m, n) == (0, 0):
            return self.K[2]
        elif (m, n) == (1, 0):
            return np.array([[0, 0],
                             [0, 2 * p['m2'] * l2]], dtype=complex)
        elif (m, n) == (2, 0):
            return np.array([[0, 0],
                             [0, 2 * p['m2']]], dtype=complex)
        # if (m, n) != (0, 0) return 0 because M is constant
        else:
            return 0

    def _dC(self, m, n):
        """Define the sucessive derivative of the $C$ matrix with respect to nu.

        Parameters
        ----------
        m, n  : int
            The order of derivation for nu[0] and nu[1].

        Returns
        -------
        : Matrix (petsc or else)
            The n-derivative of global K1 matrix
        """
        # if (m, n)=(0,0) return M
        if (m, n) == (0, 0):
            return self.K[1]
        elif (m, n) == (0, 1):
            return np.array([[0, 0],
                             [0, 1.]], dtype=complex)
        else:
            return 0


def main(plot=False):
    """Find the EP3 of the pendulum problem.

    Return
    ------
    C : CharPol
        The partial characteristic polynomial.
    sol : np.array
        Fhe EP2 solutions. For each solution, it constains the eigenvalue and l2.
    error : float
        The global error between the found EP2 and reference solution.
    """
    # Define the order of derivation for each parameter
    Nderiv = (5, 5)
    # Define the inital value of parameter
    nu0 = np.array((0.6, 0.11))
    # Instiate the problem
    pendulum = Pendulum(nu0)
    pendulum.createSolver(pb_type='PEP')
    # Run the eigenvalue computation
    Lambda = pendulum.solver.solve(target=0+0j)
    # Create a list of the eigenvalue to monitor
    lda_list = np.arange(0, 4)
    # Return the eigenvalue and eigenvector in a list of Eig object
    extracted = pendulum.solver.extract(lda_list)
    # destroy solver (important for petsc/slepc)
    pendulum.solver.destroy()
    # Compute the eigenvalue derivatives
    for vp in extracted:
        vp.getDerivativesMV(Nderiv, pendulum)
    # Build Charpol
    C = ee.CharPol.from_recursive_mult(extracted)
    # Find EP2 with Charpol at fixed c2
    C1 = C.set_param(1, c2_ref1)
    # C1 = C.set_param(1, c2_ref2)
    # Use homotopy solver
    bplp, s = C1.homotopy_solve(tracktol=1e-12, finaltol=1e-13, tol_filter=-1)
    delta, sol, deltaf = C1.filter_spurious_solution(s, plot=plot, tol=1e-1)
    # Compute the error vs the 2 EP
    i1 = np.argmin(abs(sol[:, 1] - l2_ref1))
    i2 = np.argmin(abs(sol[:, 1] - l2_ref2))
    err_c = [abs(sol[i1, 1] - l2_ref1),
             abs(sol[i2, 1] - l2_ref2)]

    # Compute real EP from a single initial guess
    z0 = np.array([0.95 * lda_ref1, *(np.array((0.69+0.003j, 0.11+0.01j)))])
    z = C.lm_from_sol(z0, real_param_ep=True)
    # Check error with references solutions
    err_r1 = [abs(z[0] - lda_ref1),
              abs(z[1] - l2_ref1),
              abs(z[2] - c2_ref1)]

    # Compute real EP in a region
    s = C.iterative_solve((None,
                           (0.2, 0.8),
                           (0.08, 0.17)), Npts=2,
                          algorithm='lm', real_param_ep=True)
    # Check error with references solutions
    i1 = np.argmin(abs(s[:, 0] - lda_ref1))
    i2 = np.argmin(abs(s[:, 0] - lda_ref2))
    err_r_all = [s[i1, 0] - lda_ref1, s[i1, 1] - l2_ref1,  s[i1, 2] - c2_ref1,
                 s[i2, 0] - lda_ref2, s[i2, 1] - l2_ref2,  s[i2, 2] - c2_ref2]
    return C, sol, err_c,  norm(err_r1), norm(err_r_all)


if __name__ == '__main__':
    C, sol, err_c, err_r1, err_r_all = main(plot=False)
    print(f'Error on real valued solver {np.linalg.norm(err_r1)}.')
    print(f'Error on complex valued solver {min(err_c)}.')
