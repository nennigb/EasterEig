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
r"""
Consider the following 3 DOF system.
```
           +~~~~~~~~~~+                +~~~~~~~~~~+               +~~~~~~~~~~+
           |          |                |          |               |          |
     mu    |          |        k       |          |        k      |          |     nu
X---/\/\---|    m1    | -----/\/\----- |    m2    | -----/\/\-----|    m3    | ---/\/\---X
           |          |                |          |               |          |
           +~~~O~~~O~~+                +~~~O~~~O~~+               +~~~O~~~O~~+
```
depending of two complex parameters mu and nu.
We would like to find the EP3.

@author: bn
"""

import networkx as nx
import numpy as np
import eastereig as ee
import matplotlib.pyplot as plt
from bruteforce import bruteForceSolvePar

class Network(ee.OPmv):
    """Create a subclass of the interface class OP that describe the problem
    operator.

    Remarks
    -------
    It is not possible to have nu0 between a node and the ground!"""

    def __init__(self, nu0):
        """Initialize the problem.

        Parameters
        ----------
            nu0 : tuple
                the parameters initial Values (complex)
        """
        self.m = 1
        self.k = 1
        self.nu0 = nu0

        # assemble *base* matrices
        Mmat = self._mass()
        Kmat = self._stiff()

        # initialize OP interface
        self.setnu0(nu0)

        # # mandatory -----------------------------------------------------------
        self._lib = 'numpy'
        # # create the operator matrices
        self.K = [Kmat, Mmat]
        # # define the list of function to compute  the derivatives of each operator matrix
        self.dK = [self._dstiff, self._dmass]
        # # define the list of function to set the eigenvalue dependance of each operator matrix
        self.flda = [None, ee.lda_func.Lda]
        # # ---------------------------------------------------------------------

    # possible to add new methods
    def __repr__(self):
        """Define the object representation.
        """
        text = "Instance of Operator class {} @nu0={}."

        return text.format(self.__class__.__name__, self.nu0)

    def _mass(self):
        """ Define the a diagonal mass matrix.
        """
        m = self.m
        M = - np.array([[m, 0, 0],
                       [0, m, 0],
                       [0, 0, m]], dtype=np.complex)
        return M

    def _stiff(self):
        """Define the stifness matrix of 1D FEM with ordered nodes.
        """
        mu = self.nu0[0]
        nu = self.nu0[1]
        k = self.k
        K = np.array([[mu+k, -k, 0.],
                      [-k, 2*k, -k],
                      [0., -k, nu+k]], dtype=np.complex)
        return K

    def _dstiff(self, m, n):
        r"""Define the sucessive derivative of the $\tilde{K}$ matrix with respect to nu.

        mu -> m
        nu -> n

        Parameters
        ----------
        m, n : int
            the order of derivation
        Returns
        -------
        Kn : Matrix (petsc or else)
            The n-derivative of global K0 matrix
        """
        k = self.k
        mu, nu = self.nu0

        if (m, n) == (0, 0):
            Kn = self.K[0]
        elif (m, n) == (1, 0):
            Kn = np.zeros_like(self.K[0], dtype=np.complex)
            Kn[0, 0] = 1.
        elif (m, n) == (0, 1):
            Kn = np.zeros_like(self.K[0], dtype=np.complex)
            Kn[2, 2] = 1.
        # if (m, n) > (1, 1) return 0 because K has a linear dependancy on nu
        else:
            return 0

        return Kn

    def _dmass(self, m, n):
        """Define the sucessive derivative of the $M$ matrix with respect to (mu, nu).

        Parameters
        ----------
        m, n  : int
            the order of derivation
        Returns
        -------
        Kn : Matrix (petsc or else)
            The n-derivative of global K1 matrix
        """
        # if (m, n)=(0,0) return M
        if (m, n) == (0, 0):
            return self.K[1]
        # if (m, n) != (0, 0) return 0 because M is constant
        else:
            return 0


# %% MAIN
if __name__ == '__main__':

    import itertools as it
    from multiple import *
    from scipy.special import factorial

    Nderiv = (5, 5)
    # tuple of the inital param
    nu0 = (1., 1)

    # %% Locate EP
    net = Network(nu0)

    net.createSolver(pb_type='gen')
    # run the eigenvalue computation
    Lambda = net.solver.solve(target=0+0j)
    # create a list of the eigenvalue to monitor
    lda_list = np.arange(0, 3)
    # return the eigenvalue and eigenvector in a list of Eig object
    extracted = net.solver.extract(lda_list)
    # destroy solver (important for petsc/slepc)
    net.solver.destroy()
    print('> Eigenvalue :', Lambda)

    vp = extracted[0]
    vp.getDerivativesMV(Nderiv, net)
    # # get derivativ of allupto
    # dLambda=[]
    # for i, vp in enumerate(extracted):
    #     print('> Get derivative vp {} ...\n'.format(i))
    #     tic = time.time()  # init timer
    #     vp.getDerivatives(Nderiv, net)
    #     print("              derivative real time :", time.time() - tic)  # stop timer
    #     print(vp.dlda)
    #     dLambda.append(np.array(vp.dlda))
    #     if export:
    #         vp.export(name + '_vp_' + str(i), eigenvec=False)
