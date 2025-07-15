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
Localization of exceptional point on a 2D waveguides lined with an impedance Z
numpy version.


```
   y=h ____________________________ rigid wall / Neumann bc

           | |
          -|-|-->     oo-duct
           | |

   y=0 ~~~~~~~~~~~~Z~~~~~~~~~~~~~~ impedance bc

```

Description
------------
This problem is described in sec. 4.1 of arxiv.org/abs/1909.11579
and yield a **generalized eigenvalue problem** and the eigenvalue lda stands for the axial wavenumber

[(k**2*Mmat + Zeff*GamMat - Kmat) - lda * Mmat ]x=0

Examples
--------
>>> import numpy as np
>>> import eastereig as ee
>>> EP1, vp1 = main() # doctest: +ELLIPSIS
Instance of Operator class Znumpy @nu0=(486.198103097114+397.605679264872j) (5 dof, height=1.0)...

Get exceptional point location and check its value
>>> EP1.EP_loc[0]
(445.90479284...+356.8118109...j)

Get 10 st puiseux series coefficients of 1st EP and check their value
>>> a=EP1.a[0][0:10]
>>> a_ref= np.array([ 1.0560473476551177e+01+4.8901202919640054e+00j,  3.0421365868096456e-01-1.3955670083202301e-01j, \
           -9.1185719298740118e-03-7.7083065862426706e-03j, -2.7872607055568643e-04+5.0667274341882065e-04j, \
            2.6653671046490202e-05+4.8563014967076544e-06j, -1.0990762138418922e-07-1.2816923556048538e-06j, \
           -5.6965695553112449e-08+2.3643628974851434e-08j,  1.8773739240147713e-09+2.2502347515985570e-09j, \
            7.4104333038139613e-11-1.1916318961767500e-10j, -6.5010455135149481e-12-1.5160308074333804e-12j])
>>> np.linalg.norm(np.abs(a - a_ref)) <1e-9
True

Check also eigenvalue derivatives
>>> dlda = vp1.dlda[0:5]
>>> dlda_ref = np.array([(8.183663044141067+4.39391090652107j), (-0.018451689720575804+0.007103750315344242j),\
    (2.7619229792202445e-05-0.00021888073028013003j), (3.938676110910742e-06+3.954262823086467e-06j), (-2.3488744896828321e-07+9.16775969596652e-09j)])
>>> np.linalg.norm(np.abs(dlda - dlda_ref)) <1e-10
True
"""


# standard
import numpy as np
import scipy as sp
from scipy.special import factorial
import matplotlib.pyplot as plt
# eastereig
import eastereig as ee


class Znumpy(ee.OP):
    """Create a subclass of the interface class OP that describe the problem operator."""

    def __init__(self, z, n, h, rho, c, k):
        """Initialize the problem.

        Parameters
        ----------
            z : complex
                impedance value
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
        self._gam()  # create _GamMat
        # initialize OP interface
        self.setnu0(z)

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
        """Define the object representation."""
        return "Instance of Operator class {} @nu0={} ({} dof, height={})".format(self.__class__.__name__,
                                                                                  self.nu0, self.n, self.h)

    def _mass(self):
        """Define the mass matrix, of 1D FEM with ordered nodes.

        The matrix is created by converting a sparse matrix into full matrix
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
        M = sp.sparse.diags([m1, m2, m1], [-1, 0, 1], shape=(n, n)).toarray()
        # Boundary points
        M[0, 0] = m2/2.
        M[0, 1] = m1
        M[n-1, n-2] = m1
        M[n-1, n-1] = m2/2.

        # store it
        self._Mmat = M

    def _stif(self):
        """Define the stifness matrix of 1D FEM with ordered nodes.

        The matrix is created by converting a sparse matrix into full matrix
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
        K = sp.sparse.diags([k1, k2, k1], [-1, 0, 1], shape=(n, n)).toarray()
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
        The matrix is created by converting a sparse matrix into full matrix

        Parameters
        ----------
        z0 : complex
            The impedance value
        """
        n = self.n
        # create Gamma matrix (complex)
        # Striffness matrix
        Gam = sp.sparse.coo_matrix((np.array([1.]), (np.array([0]), np.array([0]))), shape=(n, n)).toarray()
        # Store it
        self._GamMat = Gam

    def _ImpMat(self):
        """Return the operator matrices list for the generalized eigenvalue problem.

        Returns
        -------
        K : matrix
           K contains [k0^2*M + Gamma*1i*const.rho0*omega/Z - K , -M]
        """
        omega = self.k * self.c    # angular freq.
        Zeff = 1j * omega * self.rho / self.nu0

        K = []
        KK = self.k**2*self._Mmat + Zeff*self._GamMat - self._Kmat
        # encapsulated into list
        K.append(KK)
        K.append(-self._Mmat)

        return K

    def _dstiff(self, n):
        r"""Define the sucessive derivative of the $\tilde{K}$ matrix with respect to nu.

        L = \tilde{K}- lda M (standard FEM formalism)
        with polynomial formlism L = K0 + lda K1 + ...
        thus K0=\tilde{K}

        Parameters
        ----------
        n : int
            the order of derivation
        Returns
        -------
        Kn : Matrix (petsc or else)
            The n-derivative of global K0 matrix
        """
        if n == 0:
            Kn = self.K[0]
        else:
            omega = self.k * self.c    # angular freq.
            # derivative of (1/Z)
            Zn = (-1)**n * factorial(n)/self.nu0**(n+1)
            Kn = self._GamMat * 1j*self.rho*omega*Zn

        return Kn

    def _dmass(self, n):
        """Define the sucessive derivative of the $M$ matrix with respect to nu.

        L = K - lda M (standard FEM formalism)
        with polynomial formlism L = K0 + lda K1 + ...
        thus K1=-M

        Parameters
        ----------
        n : int
            the order of derivation
        Returns
        -------
        Kn : Matrix (petsc or else)
            The n-derivative of global K1 matrix
        """
        # if n=0 return M
        if n == 0:
            return -self._Mmat
        # if n!= 0 return 0 because M is constant
        else:
            return 0


def main(N=5):
    """Run the example.

    Parameters
    ----------
    N : integer
        number of degree of freedom

    Returns
    -------
    EP1 : EP
        The EP object containing all EP information
    vp1 : Eig
        The object containing all information on the 1st eigenvalue
    """
    import numpy as np
    import time

    pi = np.pi
    # Number of derivative
    Nderiv = 12
    # air properties
    rho0, c0 = 1.2, 340.
    # initial imepdance guess
    # z0=400+3j
    z0 = 486.198103097114 + 397.605679264872j
    # freq. of computation
    f = 200
    # modal index pair
    n1 = 0
    n2 = 1
    # number of dof
    # N=5 -> move to function
    # duct height
    h = 1.
    # number of mode to compute
    Nmodes = 4
    # solve the problem
    omega = 2*pi*f
    k0 = omega/c0
    # Create discrete operator of the pb
    Imp = Znumpy(z=z0, n=N, h=h, rho=rho0, c=c0, k=k0)
    print(Imp)
    # initialize eigenvalue solver for *generalized eigenvalue problem*
    Imp.createSolver(pb_type='gen')
    # run the eigenvalue computation
    Lambda = Imp.solver.solve(nev=Nmodes, target=0+0j, skipsym=False)
    # return the eigenvalue and eigenvector in a list of Eig object
    extracted = Imp.solver.extract([n1, n2])
    # destroy solver (important for petsc/slepc)
    Imp.solver.destroy()
    print('> Eigenvalue :', Lambda)

    print('> Get eigenvalues derivatives ...\n')
    # Create Eig object
    vp1, vp2 = extracted

    print('> Get derivative vp1 ...\n')
    tic = time.time()  # init timer
    vp1.getDerivatives(Nderiv, Imp)
    print("              derivative real time :", time.time()-tic)
    print(vp1.dlda)
    tic = time.time()  # init timer
    vp2.getDerivatives(Nderiv, Imp)
    print("              derivative real time :", time.time()-tic)
    print(vp2.dlda)

    # Locate EP
    tic = time.time()  # init timer
    # create EP instance to find the merging of vp1 and vp2
    EP1 = ee.EP(vp1, vp2)
    loc = EP1.locate()[0]

    print('\n> EP location :')
    # print EP summary
    print(EP1)
    print("              # LOC real time :", time.time()-tic)

    # compute puiseux series coefficients
    EP1.getPuiseux()

    return EP1, vp1

# Reference values from 10.1016/j.jcp.2020.109425 (using many dof)
z_ref = [445.803 + 357.211j, 246.057 + 94.9156j, 165.134 + 44.1474j, 123.660 + 25.7198j]

if __name__ == '__main__':
    """Show graphical outputs and reconstruction examples.
    """
    import numpy as np
    from matplotlib import pyplot as plt

    EP1, vp1 = main()
    # plot roots of Th to check EP localization
    EP1.plotZeros()

    # compute reconstruction
    N = 5  # limit series order to 5 terms
    points = np.linspace(-125, 125, 101) + EP1.nu0
    tay = vp1.taylor(points, n=N)
    pad = vp1.pade(points, n=N)
    pui1, pui2 = vp1.puiseux(EP1, points, n=N)
    ana1, ana2 = vp1.anaAuxFunc(EP1, points, n=N)

    # plotting
    plt.figure()
    plt.plot(points.real, tay.real, 'k', label='Taylor series')
    plt.plot(points.real, pui2.real, 'b', label='Puiseux series')  # converge well if N=10
    plt.plot(points.real, pad.real, 'teal', label='Padé appox.')
    plt.plot(points.real, ana2.real, 'g.-', label='AAF approx')
    plt.legend()
