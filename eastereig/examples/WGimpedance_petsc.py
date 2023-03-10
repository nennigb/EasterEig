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
petsc/slepc version.


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
Instance of Operator class Zpetsc @nu0=(486.198103097114+397.605679264872j) (5 dof, share with ... proc., height=1.0)...

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
>>> np.linalg.norm(np.abs(a - a_ref)) <1e-8
True

Check also eigenvalue derivatives
>>> dlda = EP1._vp1dlda[0:5]
>>> dlda_ref = np.array([(8.183663044141067+4.39391090652107j), (-0.018451689720575804+0.007103750315344242j),\
    (2.7619229792202445e-05-0.00021888073028013003j), (3.938676110910742e-06+3.954262823086467e-06j), (-2.3488744896828321e-07+9.16775969596652e-09j)])
>>> np.linalg.norm(np.abs(dlda - dlda_ref)) <1e-8
True

To run an example with petsc in parallel, you need to run python with `mpirun`. For instance, to run this example with 2 proc:
>>> import subprocess  # doctest: +SKIP
>>> out=subprocess.run("mpirun -n 2 python3 ./eastereig/examples/WGimpedance_petsc.py".split())  # doctest: +SKIP
>>> out.returncode # doctest: +SKIP
0

"""
# -*- coding: utf-8 -*-
from __future__ import print_function, division
from scipy.special import factorial
import numpy as np
import eastereig as ee
from petsc4py import PETSc
import sys
import slepc4py
slepc4py.init(sys.argv)

Print = PETSc.Sys.Print


class Zpetsc(ee.OP):
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
        self._lib = 'petsc'
        # create the operator matrices
        self.K = self._ImpMat()
        # define the list of function to compute  the derivatives of each operator matrix
        self.dK = [self._dstiff, self._dmass]
        # define the list of function to set the eigenvalue dependance of each operator matrix
        self.flda = [None, ee.lda_func.Lda]
        # ---------------------------------------------------------------------

    # Possible to add new methods
    def __repr__(self):
        """Define the object representation."""
        return "Instance of Operator class {} @nu0={} ({} dof, share with {} proc., height={})".format(self.__class__.__name__,
                                                                                                       self.nu0, self.n,
                                                                                                       PETSc.COMM_WORLD.getSize(),
                                                                                                       self.h)

    def _mass(self):
        """Define the mass matrix, of 1D FEM with ordered nodes.

        The elementary matrix reads
        Me = (Le/6) * [2  1; 1 2]
        Thus the lines are [2,1,0], [1,4,1], [0,1,2] x (Le/6)
        """
        n = self.n
        # create M matrix (complex)
        M = PETSc.Mat().create()
        M.setSizes([n, n])
        M.setType('aij')
        M.setPreallocationNNZ(3)
        M.setUp()

        Istart, Iend = M.getOwnershipRange()
        i1 = Istart
        if Istart == 0:
            i1 = i1 + 1  # modify outside loop
        i2 = Iend
        if Iend == n:
            i2 = i2 - 1   # modify outside loop

        # value for a inner M row [m1 m2 m1]
        m1 = self.Le / 6.
        m2 = self.Le * 4./6.
        # Interior grid points
        for i in range(i1, i2):
            M[i, i-1] = m1
            M[i, i] = m2
            M[i, i+1] = m1

        # Boundary points
        if Istart == 0:
            M[0, 0] = 2.*m1
            M[0, 1] = m1
        if Iend == n:
            M[n-1, n-2] = m1
            M[n-1, n-1] = 2.*m1
        # petsc will assemble the matrix
        M.assemble()

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
        K = PETSc.Mat().create()
        K.setSizes([n, n])
        K.setType('aij')
        K.setPreallocationNNZ(3)
        K.setUp()

        # value for inner K row [k1 k2 k1]
        k1 = -1. / self.Le
        k2 = 2. / self.Le

        # Striffness matrix
        Istart, Iend = K.getOwnershipRange()
        i1 = Istart
        if Istart == 0:
            i1 = i1 + 1  # modify outside loop
        i2 = Iend
        if Iend == n:
            i2 = i2 - 1   # modify outside loop

        # Interior grid points
        for i in range(i1, i2):
            K[i, i-1] = k1
            K[i, i] = k2
            K[i, i+1] = k1

        # Boundary points
        if Istart == 0:
            K[0, 0] = k2/2.
            K[0, 1] = k1
        if Iend == n:
            K[n-1, n-2] = k1
            K[n-1, n-1] = k2/2.
        # petsc will assemble the matrix
        K.assemble()
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
        # create K and M matrix (complex)
        Gam = PETSc.Mat().create()
        Gam.setSizes([n, n])
        Gam.setType('aij')
        Gam.setPreallocationNNZ(1)
        Gam.setUp()

        # Striffness matrix
        """
        https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatAssemblyBegin.html#MatAssemblyBegin
        Space for preallocated nonzeros that is not filled by a call to MatSetValues() or a related routine are compressed out by assembly.
        ie the pre allocted non initialized values are removed...
        """
        Istart, Iend = Gam.getOwnershipRange()
        if Istart == 0:
            Gam[0, 0] = 1.
        # petsc will assemble the matrix
        Gam.assemble()
        # store it
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

    import numpy as np
    import time

    # mpi communicator
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
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
    Imp = Zpetsc(z=z0, n=N, h=h, rho=rho0, c=c0, k=k0)
    print(Imp)
    Imp.createSolver(pb_type='gen')
    Lambda = Imp.solver.solve(nev=Nmodes, target=0+0j, skipsym=False)
    # return the eigenvalue and eigenvector in a list of Eig object
    extracted = Imp.solver.extract([n1, n2])
    # destroy solver (important for petsc/slepc)
    Imp.solver.destroy()
    print('> Eigenvalue :', Lambda)

    print('> Get eigenvalues derivatives ...\n')
    # Create Eig object
    vp1, vp2 = extracted

    Print('> Get derivative vp1 ...\n')
    tic = time.time()  # init timer
    vp1.getDerivatives(Nderiv, Imp)
    Print("              derivative real time :", time.time()-tic)
    Print(vp1.dlda)
    tic = time.time()  # init timer
    vp2.getDerivatives(Nderiv, Imp)
    Print("              derivative real time :", time.time()-tic)
    Print(vp2.dlda)

    # Locate EP
    EP1 = ee.EP(vp1, vp2)
    if rank == 0:
        tic = time.time()  # init timer
        loc = EP1.locate()[0]
        # print EP summary
        print(EP1)
        # compute puiseux series coefficients
        EP1.getPuiseux()
    else:
        EP1 = None
    # share EP1 between all node
    EP1 = comm.bcast(EP1, root=0)
    # print(EP1.EP_loc)
    return EP1, vp1


if __name__ == '__main__':
    """Show graphical outputs and reconstruction examples."""

    EP1, vp1 = main()
    # comm=PETSc.COMM_WORLD.tompi4py()
    # rank = comm.Get_rank()

    # compute reconstruction
    N = 5  # limit series order to 5 terms
    points = np.linspace(-125, 125, 101) + EP1.nu0
    tay = vp1.taylor(points, n=N)
    pad = vp1.pade(points, n=N)
    pui1, pui2 = vp1.puiseux(EP1, points, n=N)
    ana1, ana2 = vp1.anaAuxFunc(EP1, points, n=N)
