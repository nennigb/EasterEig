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
Localization of exceptional point on a 2D waveguides lined with two adminttances
numpy version.
Resolution use the admintance Y=1/Z to avoid singularities with the impedance Z.

This allow to find EP3.
```
   y=h ____________________________ admintance bc (nu)

           | |
          -|-|-->     oo-duct
           | |

   y=0 ~~~~~~~~~~~~Y~~~~~~~~~~~~~~ admintance bc (mu)

```

Description:
------------
This problem is described in sec. 4.1 of arxiv.org/abs/1909.11579
and yield a **generalized eigenvalue problem** and the eigenvalue lda stands
for the axial wavenumber. The admintance is Y=1/Z.

[(k**2*Mmat + Yeff*GamMat - Kmat) - lda * Mmat ]x=0

FIXME : put good value in the example

Examples
--------
>>> import numpy as np
>>> import eastereig as ee
>>> EP1, vp1 = main() # doctest: +ELLIPSIS
Instance of Operator class Ynumpy @nu0=(0.0012325073357595624-0.0010079264261048232j) (5 dof, height=1.0)...

Get exceptional point location and check its value
>>> EP1.EP_loc[0]
(0.001367...-0.001094...j)

"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# standard
import sys
import numpy as np
import scipy as sp
from scipy.special import factorial
import matplotlib.pyplot as plt
# eastereig
import eastereig as ee


class Ynumpy(ee.OPmv):
    """Create a subclass of the interface class OP that describe the problem operator."""
    def __init__(self, y, n, h, rho, c, k):
        """Initialize the problem.

        Parameters
        ----------
            y : iterable
                admintance values (mu, nu)
            n : int
                number of dof
            h : float
                length of the cavity
            rho : float
                air density
            c : float
                air sound speed
            k : float
                air wavenumber
        """

        self.h = float(h)
        self.n = n
        # air properties
        self.rho, self.c, self.k = rho, c, k
        # element length
        self.Le = h/(n-1)

        # assemble *base* matrices, independant of the parameter
        self._mass()  # create _Mmat
        self._stif()  # create _Kmat
        self._gam()   # create _GamMat
        # initialize OP interface
        self.setnu0(y)

        # mandatory -----------------------------------------------------------
        self._lib = 'numpy'
        # create the operator matrices
        self.K = self._ImpMat()
        # define the list of function to compute  the derivatives of each operator matrix
        self.dK = [self._dstiff, self._dmass]
        # define the list of function to set the eigenvalue dependance of each operator matrix
        self.flda = [None, ee.lda_func.Lda]
        # ---------------------------------------------------------------------

    # possible to add new methods
    def __repr__(self):
        """ Define the object representation
        """
        return "Instance of Operator class {} @nu0={} ({} dof, height={})".format(self.__class__.__name__, self.nu0, self.n, self.h)

    def _mass(self):
        """ Define the mass matrix, of 1D FEM with ordered nodes.

        The elementary matrix reads
        Me = (Le/6) * [2  1; 1 2]
        Thus the lines are [2,1,0], [1,4,1], [0,1,2] x (Le/6)
        """
        n = self.n
        # create M matrix (complex)
        # value for a inner M row [m1 m2 m1]
        m1 = self.Le / 6.
        m2 = self.Le * 4./6.
        # Interior grid points
        M = sp.sparse.diags([m1, m2, m1], [-1, 0, 1],
                            shape=(n, n), format='csc').toarray()
        # Boundary points
        M[0, 0] = m2/2.
        M[0, 1] = m1
        M[n-1, n-2] = m1
        M[n-1, n-1] = m2/2.

        # store it
        self._Mmat = M

    def _stif(self):
        """Define the stifness matrix of 1D FEM with ordered nodes.

        The elementary matrix read
        Ke = (1/Le) * [1 -1; -1 1]
        Thus the lines are [1,-1,0], [-1,2,-1], [0,-1,1] x (1/Le)
        """
        n = self.n
        # create K and M matrix (complex)
        # value for inner K row [k1 k2 k1]
        k1 = -1. / self.Le
        k2 = 2. / self.Le

        # Striffness matrix
        # Interior grid points
        K = sp.sparse.diags([k1, k2, k1], [-1, 0, 1], shape=(n, n), format='csc').toarray()
        # Boundary points
        K[0, 0] = k2/2.
        K[0, 1] = k1
        K[n-1, n-2] = k1
        K[n-1, n-1] = k2/2.

        # store it
        self._Kmat = K

    def _gam(self):
        r"""Define the Gamma matrix accounting for the impedance BC.

        Zeros everywhere except on the 1st node $\Gamma(0) = 1 $

        Parameters
        ----------
        z0 : complex
            The impedance value
        """
        n = self.n
        # create Gamma matrix (complex)
        # Striffness matrix
        Gam_top = sp.sparse.coo_matrix((np.array([1.]), (np.array([0]), np.array([0]))),
                                   shape=(n, n)).toarray()
        Gam_bot = sp.sparse.coo_matrix((np.array([1.]), (np.array([n-1]), np.array([n-1]))),
                                    shape=(n, n)).toarray()
        # Store it
        self._GamMat = {'mu': Gam_bot, 'nu': Gam_top} # mu, nu

    def _ImpMat(self):
        """Return the operator matrices list for the generalized eigenvalue problem.

        Returns
        -------
        K : matrix
           K contains [k0^2*M + Gamma*1i*const.rho0*omega/Z - K , -M]
        """
        omega = self.k * self.c    # angular freq.
        mu = 1j * omega * self.rho * self.nu0[0]
        nu = 1j * omega * self.rho * self.nu0[1]

        K = []
        KK = self.k**2*self._Mmat + mu*self._GamMat['mu'] + nu*self._GamMat['nu'] - self._Kmat
        # encapsulated into list
        K.append(KK)
        K.append(-self._Mmat)

        return K

    def _dstiff(self, m, n):
        """ Define the sucessive derivative of the $\tilde{K}$ matrix with respect to nu

        L = \tilde{K}- lda M (standard FEM formalism)
        with polynomial formlism L = K0 + lda K1 + ...
        thus K0=\tilde{K}

        Parameters
        ----------
        m, n : int
            the order of derivation for each variable.

        Returns
        -------
        Kn : Matrix (petsc or else)
            The n-derivative of global K0 matrix
        """
        if (m, n) == (0, 0):
            Kn = self.K[0]
        elif (m, n) == (1, 0):
            omega = self.k * self.c    # angular freq.
            Kn = self._GamMat['mu'] * 1j*self.rho*omega
        elif (m, n) == (0, 1):
            omega = self.k * self.c    # angular freq.
            Kn = self._GamMat['nu'] * 1j*self.rho*omega
        else:
            Kn = 0

        return Kn

    def _dmass(self, m, n):
        """Define the sucessive derivative of the $M$ matrix with respect to nu.

        L = K - lda M (standard FEM formalism)
        with polynomial formlism L = K0 + lda K1 + ...
        thus K1=-M

        Parameters
        ----------
        m, n : int
            the order of derivation for each variable.
        Returns
        -------
        Kn : Matrix (petsc or else)
            The n-derivative of global K1 matrix
        """
        # if (m, n) = 0 return M, else 0
        if (m, n) == (0, 0):
            return -self._Mmat
        # if n!= 0 return 0 because M is constant
        else:
            return 0


# def main(N=5):


""" run the example

Parameters
-----------
N : integer
    number of degree of freedom

Returns
-------
EP1 : EP
    The EP object containing all EP information
vp1 : Eig
    The object containing all information on the 1st eigenvalue
"""
N = 100
pi = np.pi
# Number of derivative
Nderiv = (4, 4)
# wavenumer and air properties
rho0, c0, k0 = 1., 1., 1.
# duct height
h = 1.
# initial imepdance guess
# nu0 = [3.0 + 3j, 3.0 + 3.0j]
# nu0 = [0.0+0.00j, 0.]
nu0_manu = np.array([3.1781+ 4.6751j, 3.0875 + 3.6234j])
nu0 = 1.5*nu0_manu/1j

# number of dof
# N=5 -> move to function

# number of mode to compute
Nmodes = 5

# Create discrete operator of the pb
imp = Ynumpy(y=nu0, n=N, h=h, rho=rho0, c=c0, k=k0)
print(imp)
# Initialize eigenvalue solver for *generalized eigenvalue problem*
imp.createSolver(pb_type='gen')
# run the eigenvalue computation
Lambda = imp.solver.solve(nev=Nmodes, target=0+0j, skipsym=False)
# create a list of the eigenvalue to monitor
lda_list = np.arange(0, 12)
# return the eigenvalue and eigenvector in a list of Eig object
extracted = imp.solver.extract(lda_list)
# destroy solver (important for petsc/slepc)
imp.solver.destroy()
print('> Eigenvalue :', Lambda)
print('> alpha/pi :', np.sqrt(k0 - Lambda)/np.pi)

print('> Get eigenvalues derivatives ...\n')
for vp in extracted:
    vp.getDerivativesMV(Nderiv, imp)

C = ee.CharPol(extracted)

# From analyical solution:
alpha_s = 4.1969 - 2.6086j
lda_s = k0**2 - alpha_s**2
nu_s = np.array((3.1781+ 4.6751j, 3.0875 + 3.6234j))/ 1j

val_s = np.array((lda_s, *nu_s))
C.eval_at(val_s)

val0 = np.array((extracted[0].lda, *nu0))
# Locate (lda, EP)
sol = C.newton(((lda_s*0.9, lda_s*4.),
                (2+2j, 4+8j),
                (2+2j, 4+8j)), decimals=3, tol=1e-6, Npts=7)
# Best stragety is 1st run coarse and second run to refine.
# sort roots by nu_i modulus
# expression solution in alpha
sol[:,0] = np.sqrt(k0**2 - sol[:,0])
s = C._newton(C.EP_system, C.jacobian, val0, tol=1e-4, verbose=True)
    # return imp, sol

# if __name__=='__main__':
#     """Show graphical outputs and reconstruction examples."""
#     imp, sol = main()
