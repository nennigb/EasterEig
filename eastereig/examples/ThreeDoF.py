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
Analytic reconstruction of the eigenvalue loci of a three degrees of freedom
toy model when one if its stiffness parameter is random or varies on a given range.

This script aims to study the performances of the Taylor, Puiseux expansions and
a new approach based on analytic auxilliary functions.
In particular, the performances of these approximations are studied in relation
with the veering phenomenon.

The influence of the veering phenomenon is studied by varying the k_3 parameter
of the system. This script allows to consider all the other parameters of the
3-dof system.

```
           +~~~~~~~~~~+               nu                +~~~~~~~~~~+
           |          | -------------/\/\-------------- |          |
           |    m1    |                                 |    m3    |
     k1    |          |          +~~~~~~~~~~+           |          |     k3
X---/\/\---|          |---/\/\---|          | ---/\/\---|          | ---/\/\---X
           +~~~~~~~~~~+    k4    |    m2    |     k5    +~~~~~~~~~~+
                                 |          |
                      X---/\/\---|          |
                           k2    +~~~~~~~~~~+
```

Description
------------
This problem is described in a paper actually under review
and yield a **generalized parametric eigenvalue problem**.
The eigenvalue lda stands for the square of the resonance frequency and nu is
the varying parameter of this problem

[Kmat(nu) - lda(nu) * Mmat ]x(nu)=0

Examples
--------
>>> import numpy as np
>>> import eastereig as ee
>>> import scipy as sp
>>> # Varying stiffness
>>> k6_0 = 1. +0.j
>>> k6_idx = 5
>>> k_0=[1.,2.,3,1.,1.,k6_0]
>>> EPs, evs = myMain(k6_0, k6_idx,k_0,Nderiv=12)  # doctest: +ELLIPSIS
Instance of Operator class ThreeDof @nu0=(1+0j) (3 dof, k=[1.+0.j 2.+0.j 3.+0.j 1.+0.j 1.+0.j 1.+0.j])...

Check eigenvalues
>>> ev0,ev1,ev2 = evs
>>> abs(ev0.lda - (1.78568026+0.j))<1e-6
True

Get exceptional points location and check values
>>> EP1, EP2 = EPs
>>> EP1.locate()
[]
>>> res_EP2 = np.sort_complex(EP2.locate())[0]    # since cc
>>> res_EP2
(0.892616079814...-0.597704202996...j)

Check eigenvalue reconstruction with Taylor, Padé, Puiseux and Analytic auxiliary functions
>>> N = 6
>>> check_pt = np.asarray(EP2.nu0 + 0.5*res_EP2.real)
>>> k_0[5]=check_pt
>>> model = ThreeDof(k_0,check_pt,k6_idx)
>>> model.createSolver(pb_type='gen')
>>> lda_ = model.solver.solve()
> Solve gen eigenvalue problem with NumpyEigSolver class...
<BLANKLINE>
>>> tay2 = ev2.taylor(check_pt,n=N)
>>> pad2 = ev2.pade(check_pt,n=N)
>>> pui2,pui1 = ev1.puiseux(EP2,check_pt,n=N)
>>> aaf2,aaf1 = ev1.anaAuxFunc(EP2,check_pt,n=N)
>>> abs(tay2 - lda_[2])/abs(lda_[2]) < 1e-3
True
>>> abs(pad2 - lda_[2])/abs(lda_[2]) < 1e-3
True
>>> abs(pui2 - lda_[2])/abs(lda_[2]) < 1e-2
True
>>> abs(aaf2 - lda_[2])/abs(lda_[2]) < 1e-3
True

"""

# standard
import numpy as np
import matplotlib.pyplot as plt
# eastereig
import eastereig as ee


class ThreeDof(ee.OP):
    """Create a subclass of the class `OP` that describes the operator of the problem."""

    def __init__(self, k0, nu0, idx_nu):
        """Initialize the problem.

        The following data must be provided to properly initialize the operator class :
            -> The operator matrices
            -> The list of functions defining the derivative of the operator
            matrices with respect to the parameter
            -> The list of function defining the dependancy of the operator
            matrices to the eigenvalue

        Parameters
        ----------
        k0: array
            Array with all spring stiffness.
        nu0: complex
            The parameter value where the computation are performed.
        idx_nu: int
            The position of the parameter among the springs.
        """
        # mass parameters and mass matrix
        self.m_0 = [1., 1., 1.]
        self._Mmat = self._mass()  # create Mmat

        # stiffness parameters and stiffness matrix
        self.k_0 = np.copy(k0)
        self.k_0[idx_nu] = nu0
        self.idx_nu = idx_nu
        self._Kmat = self._stiff(self.k_0)  # create Kmat

        # initialize OP interface
        self.setnu0(nu0)

        # mandatory  -----------------------------------------------------------
        self._lib = 'numpy'  # The library used to compute
        # create the operator matrices
        self.K = [self._Kmat, -1*self._Mmat]
        # define the list of function to compute  the derivatives of each operator matrix
        self.dK = [self._dmat0, self._dmat1]
        # define the list of function to set the eigenvalue dependance of each operator matrix
        self.flda = [None, ee.lda_func.Lda]
        #  ---------------------------------------------------------------------

    def _mass(self):
        """Define the mass matrix of the 3 dof system."""
        M = np.array([[self.m_0[0], 0., 0.],
                      [0., self.m_0[1], 0.],
                      [0., 0., self.m_0[2]]], dtype=complex)
        return M

    def _stiff(self, k):
        """Define the stiffness matrix of the 3 dof system."""
        K = np.array([[k[0]+k[3]+k[5], -k[3], -k[5]],
                      [-k[3], k[1]+k[3]+k[4], -k[4]],
                      [-k[5], -k[4], k[2]+k[4]+k[5]]], dtype=complex)
        return K

    def _dmat1(self, n):
        r"""Define derivative with respect to nu of the $\tilde{M}$ matrix.

        N.B. : The prototype of this function is fixed, the n parameter
        corresponds to the order of derivative. If the derivative is null,
        the function have to return a 0-valued integer.

        Usually, with a standard FEM formalism
            L = K - lda M

        For the sake of generality, a polynomial formalism is preferred
            L = K0 + lda K1 + ...
        then K1=-M

        Parameters
        ----------
        n : int
            the order of derivation

        Returns
        -------
        K1 : array_like, matrix
            The n-derivative of global K1 matrix (here the mass matrix)
        """
        # if n=0 return M
        if n == 0:
            return -self._Mmat
        # if n!= 0 return 0 because M is constant
        else:
            return 0

    def _dmat0(self, n):
        """Define derivative with respect to nu of the $\tilde{K}$ matrix.

        N.B. : The prototype of this function is fixed, the n parameter
        corresponds to the order of derivative. If the derivative is null,
        the function have to return a 0-valued integer.

        Usually, with a standard FEM formalism
            L = K - lda M

        For the sake of generality, a polynomial formalism is preferred
            L = K0 + lda K1 + ...
        then K0=K

        Parameters
        ----------
        n : int
            the order of derivation
        idx_nu : int
            index of the component of k_0 corresponding to the varying parameter

        Returns
        -------
        K0 : array_like, Matrix
            The n-derivative of global K0 matrix (here the stiffness matrix)
        """
        # if n=0 return M
        if n == 0:
            return self._Kmat
        elif n == 1:
            k_deriv = np.zeros(np.shape(self.k_0))
            k_deriv[self.idx_nu] = 1.
            return self._stiff(k_deriv)
        # if n>1 return 0 because K has a linear dependancy on nu
        else:
            return 0

    # add new methods

    def __repr__(self):
        """Define the object representation."""

        return "Instance of Operator class {} @nu0={} ({} dof, k={})".format(self.__class__.__name__,
                                                                             self.nu0, 3, self.k_0)


def myMain(k6_0, k6_idx, k_0, Nderiv=12):
    """Run the example.

    Parameters
    ----------
    k_0 : (6,) array_like
        vector of the system stiffnesses
    Nderiv: int (default: 12)
        order of derivative
        (because 12 derivatives is generally good enough)

    Returns
    -------
    EPs : list(EP)
        A list of EP objects. Each EP containing all the informations associated
        to the two EP of this 3dof problem.
    vps : list(Eig)
        A list of three Eig objects. Each Eig object is associated to one of
        the three eigenvalue of the system
    """
    # Create the discrete operator associate with the studied ThreeDof system
    toyModel = ThreeDof(k_0, k6_0, k6_idx)
    print(toyModel)
    # Initialize the eigenvalue solver
    toyModel.createSolver(pb_type='gen')
    # Run eigenvalue solver
    Lambda = toyModel.solver.solve()
    print('> Eigenvalue :', Lambda)
    # return the eigenvalue and eigenvector in a list of Eig object
    extracted_eV = toyModel.solver.extract([0, 1, 2])

    print('> Get eigenvalues derivatives  ...\n')
    # Instanciate Eig objects (by depacking extracted_eV in ev0,ev1,ev2)
    ev0, ev1, ev2 = extracted_eV

    # Then compute the eigenvalues derivatives
    # N.B. : Eigenderivatives have to be computed before trying to locate EPs
    ev0.getDerivatives(Nderiv, toyModel)
    ev1.getDerivatives(Nderiv, toyModel)
    ev2.getDerivatives(Nderiv, toyModel)

    print('> Locate EPs :')
    # create EP instances to find the merging between eigenvalues

    # Between ev0 and ev1
    EP1 = ee.EP(ev0, ev1)
    res_EP1 = EP1.locate()
    if not res_EP1:
        print("     can't find EP1...")
    else:
        print('     EP1 =', res_EP1[0])

    # Between ev0 and ev1
    EP2 = ee.EP(ev1, ev2)
    res_EP2 = EP2.locate()
    if not res_EP2:
        print("     can't find EP2...")
    else:
        print('     EP2 =', res_EP2[0])

    # Puiseux
    if res_EP1:
        EP1.getPuiseux()

    if res_EP2:
        EP2.getPuiseux()

    return [EP1, EP2], extracted_eV


if __name__ == '__main__':
    """Illustrate the basic features of the ThreeDof model, with some plots."""
    import numpy as np
    import scipy as sp

    # Varying stiffness
    k6_0 = 1. + 0.j
    k6_idx = 5
    k_0 = [1., 2., 3, 1., 1., k6_0]

    EPs, evs = myMain(k6_0, k6_idx, k_0, Nderiv=12)

    # depacking
    EP1, EP2 = EPs
    res_EP1 = EP1.locate()
    res_EP2 = EP2.locate()
    ev0, ev1, ev2 = evs

    print('> Eigenvalues reconstruction')
    # compute reconstruction
    # limit series order
    N = 6

    points = np.linspace(-1, 1, 101) + EP1.nu0
    # Taylor
    tay0 = ev0.taylor(points, n=N)
    tay1 = ev1.taylor(points, n=N)
    tay2 = ev2.taylor(points, n=N)
    # Padé
    pad0 = ev0.pade(points, n=N)
    pad1 = ev1.pade(points, n=N)
    pad2 = ev2.pade(points, n=N)
    # Puiseux and Analytic auxiliary functions
    if res_EP1:
        pui1, pui0 = ev0.puiseux(EP1, points, n=N)
        aaf1, aaf0 = ev0.anaAuxFunc(EP1, points, n=N)
    if res_EP2:
        pui2, pui1 = ev1.puiseux(EP2, points, n=N)
        aaf2, aaf1 = ev1.anaAuxFunc(EP2, points, n=N)

    # plot roots of T_h (EP1)
    f1 = plt.figure()
    EP1.plotZeros(fig=f1.number)
    # plot roots of T_h (EP2)
    f2 = plt.figure()
    EP2.plotZeros(fig=f2.number)
    # plotting
    plt.figure(figsize=(10, 5))

    plt.plot(points.real, tay0.real, 'r', label=r"$\lambda_0$'s Taylor series")
    plt.plot(points.real, tay1.real, 'g', label=r"$\lambda_1$'s Taylor series")
    plt.plot(points.real, tay2.real, 'b', label=r"$\lambda_2$'s Taylor series")

    if res_EP1:
        plt.plot(points.real, pui0.real, 'r', linestyle="--", label=r"$\lambda_0$'s Puiseux series")
        plt.plot(points.real, pui1.real, 'g', linestyle="--", label=r"$\lambda_1$'s Puiseux series")
        plt.plot(points.real, aaf0.real, 'r', linestyle="-.",
                 label=r"$\lambda_0$'s Analytic Aux. func approximation")
        plt.plot(points.real, aaf1.real, 'g', linestyle="-.",
                 label=r"$\lambda_1$'s Analytic Aux. func approximation")
    if res_EP2:
        plt.plot(points.real, pui1.real, 'g', linestyle="--", label=r"$\lambda_1$'s Puiseux series")
        plt.plot(points.real, pui2.real, 'b', linestyle="--", label=r"$\lambda_2$'s Puiseux series")
        plt.plot(points.real, aaf1.real, 'g', linestyle="-.",
                 label=r"$\lambda_1$'s Analytic Aux. func approximation")
        plt.plot(points.real, aaf2.real, 'b', linestyle="-.",
                 label=r"$\lambda_2$'s Analytic Aux. func approximation")

    plt.plot(points.real, pad0.real, 'r', linestyle=":", label=r"$\lambda_0$'s Padé approximation")
    plt.plot(points.real, pad1.real, 'g', linestyle=":", label=r"$\lambda_1$'s Padé approximation")
    plt.plot(points.real, pad2.real, 'b', linestyle=":", label=r"$\lambda_2$'s Padé approximation")

    # plt.tight_layout(rect=[0,0,0.68,1])
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel(r'Re $\nu$')
    plt.ylabel(r'Re $\lambda$')
    plt.show()

    print("> Plot Eigenvalues' Riemann surface")
    # Plot Riemann surface
    nRe, nIm = (31, 31)
    k6_0Re, k6_0Im = np.linspace(-0.5, 2.5, nRe), np.linspace(-2, 2, nIm)
    Rek6, Imk6 = np.meshgrid(k6_0Re, k6_0Im)
    k6_2D = Rek6 + 1j * Imk6

    lambda_2D = []
    for k6_ in k6_2D:
        lda2D = []
        for k6 in k6_:
            k_0[5] = k6
            model4Riemann = ThreeDof(k_0, k6, k6_idx)
            K4Riemann = model4Riemann._Kmat
            M4Riemann = model4Riemann._Mmat
            lda_, phi_ = sp.linalg.eig(K4Riemann, b=M4Riemann)
            lda2D.append(lda_)
        lambda_2D.append(lda2D)

    # Create the loci object for Riemann Surface
    loci = ee.Loci(lambda_2D, k6_2D)

    # Display Riemann surface
    loci.plotRiemann(Type='Re', N=3, EP_loc=res_EP1+res_EP2,
                     Title=r'Riemann surfaces of $\lambda_0$, $\lambda_1$ and $\lambda_2$',
                     Couleur='k', variable='\\nu', fig=-2, nooutput=False)

    check_pt = np.asarray(EP2.nu0 + 0.5*res_EP2[0].real)
    print('> Test eigenvalue reconstruction at point ', check_pt)
    k_0[5] = check_pt
    model = ThreeDof(k_0, check_pt, k6_idx)
    Ktest = model._Kmat
    Mtest = model._Mmat
    lda_test, phi_ = sp.linalg.eig(Ktest, b=Mtest)
    tay2test = ev2.taylor(check_pt, n=N)
    pad2test = ev2.pade(check_pt, n=N)
    pui2test, p1 = ev1.puiseux(EP2, check_pt, n=N)
    aaf2test, a1 = ev1.anaAuxFunc(EP2, check_pt, n=N)
    print('>    Taylor rel. error = ', abs(tay2test - lda_test[2])/abs(lda_test[2]))
    print('>    Padé rel. error = ', abs(pad2test - lda_test[2])/abs(lda_test[2]))
    print('>    Puiseux rel. error = ', abs(pui2test - lda_test[2])/abs(lda_test[2]))
    print('>    Anal. Aux. Func. rel. error = ', abs(aaf2test - lda_test[2])/abs(lda_test[2]))
