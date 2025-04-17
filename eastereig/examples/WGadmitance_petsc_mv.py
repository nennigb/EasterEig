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
Localization of exceptional point on a 2D waveguide lined with two adminttances
using petsc and slepc.
The resolution use admintance Y=1/Z to avoid the singularities present with
the impedances.

```
   y=h ____________________________ admintance bc (nu)

           | |
          -|-|-->     oo-duct
           | |

   y=0 ~~~~~~~~~~~~Y~~~~~~~~~~~~~~ admintance bc (mu)

```

Description:
------------
The analytical eigenvalue problem and references values are described in
sec. 2 of 10.1016/j.jsv.2021.116510 (available at
https://hal.science/hal-03388773v1/document).
This yields a **generalized eigenvalue problem** where the eigenvalue lda stands
for the axial wavenumber after FEM discretisation (in sec. 4.1 of
arxiv.org/abs/1909.11579). The admintance is Y=1/Z.

```
[(k**2 - lda)*Mmat + mu*iωρ*Gam_bot + nu*iωρ*Gam_top - Kmat]x=0
```

It is noteworthy that mu and nu are multiplied by 1j wrt to the references papers.

Examples
--------
>>> import numpy as np
>>> import eastereig as ee
>>> C, sol, error = main(plot=True)  # doctest: +ELLIPSIS
Start...
>>> error < 1e-3
True

To run an example with petsc in parallel, you need to run python with `mpirun`. For instance,
to run this example with 2 proc:
```console
mpirun -n 2 python3 ./eastereig/examples/WGadmittance_petsc.py
```
"""
import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
# eastereig
import eastereig as ee
from petsc4py import PETSc
Print = PETSc.Sys.Print


class Ypetsc(ee.OPmv):
    """Create a subclass of the interface class OP that describe the problem operator."""

    def __init__(self, y, n, h, rho, c, k):
        """Initialize the problem.

        Parameters
        ----------
        y : iterable
            admitance values (mu, nu)
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
        self._lib = 'petsc'
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
        """
        n = self.n
        # Initialize the matrices
        Gam_top = PETSc.Mat().create()
        Gam_top.setSizes([n, n])
        Gam_top.setType('aij')
        Gam_top.setPreallocationNNZ(1)
        Gam_top.setUp()

        Gam_bot = PETSc.Mat().create()
        Gam_bot.setSizes([n, n])
        Gam_bot.setType('aij')
        Gam_bot.setPreallocationNNZ(1)
        Gam_bot.setUp()

        # populate the matrix
        Istart, Iend = Gam_top.getOwnershipRange()
        if Istart == 0:
            Gam_top[0, 0] = 1.

        Istart, Iend = Gam_bot.getOwnershipRange()
        if Iend == n:
            Gam_bot[n-1, n-1] = 1.
        # petsc will then assemble the matrices
        Gam_top.assemble()
        Gam_bot.assemble()
        # Store it
        self._GamMat = {'mu': Gam_bot, 'nu': Gam_top}  # mu, nu

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
        r"""Define the sucessive derivative of the $\tilde{K}$ matrix with respect to nuv.

        L = \tilde{K} - lda M (standard FEM formalism)
        with polynomial formlism L = K0 + lda K1 + ...
        thus K0=\tilde{K}

        Parameters
        ----------
        m, n : int
            The order of derivation for each variable.

        Returns
        -------
        Kn : Matrix (petsc or else)
            The n-derivative of global K0 matrix
        """
        if (m, n) == (0, 0):
            Kn = self.K[0]
        elif (m, n) == (1, 0):
            omega = self.k * self.c    # angular freq.
            Kn = self._GamMat['mu'] * 1j * self.rho * omega
        elif (m, n) == (0, 1):
            omega = self.k * self.c    # angular freq.
            Kn = self._GamMat['nu'] * 1j * self.rho * omega
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
            The order of derivation for each variable.

        Returns
        -------
        Kn : Matrix (petsc or else)
            The n-derivative of global K1 matrix
        """
        if (m, n) == (0, 0):
            return -self._Mmat
        # if n!= 0 return 0 because M is constant
        else:
            return 0


# Value from 10.1016/j.jsv.2021.116510
nu_ref = np.array((3.1781 + 4.6751j, 3.0875 + 3.6234j, 3.6598 + 7.9684j,
                   3.6015 + 6.9459j, 3.9800 + 11.189j, 3.9371 + 10.176j,
                   1.0119 + 4.6029j, 1.0041 + 7.7896j)) / 1j


def main(plot=False):
    """Locate the EP3 and check the error with reference solution."""
    # mpi communicator
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()

    # Number of dof
    N = 200
    # Number of derivative
    Nderiv = (5, 5)
    # Wavenumer and air properties
    rho0, c0, k0 = 1., 1., 1.
    # Duct height
    h = 1.
    # Initial value of the admittances
    nu0 = np.array([7.01265-4.76715j, 2.89872-2.47j])

    # Number of mode to keep in the CharPol
    Nmodes = 7
    print('Start')
    # Create discrete operator of the pb
    imp = Ypetsc(y=nu0, n=N, h=h, rho=rho0, c=c0, k=k0)
    # Initialize eigenvalue solver for *generalized eigenvalue problem*
    imp.createSolver(pb_type='gen')
    # Run the eigenvalue computation
    Lambda = imp.solver.solve(nev=Nmodes, target=0+0j, skipsym=False)
    # Create a list of the eigenvalue to monitor
    lda_list = np.arange(0, Nmodes)
    # Return the eigenvalue and eigenvector in a list of Eig object
    extracted = imp.solver.extract(lda_list)
    # Destroy solver
    imp.solver.destroy()

    # Compute eigenvalues derivatives
    # This is a collective task
    for vp in extracted:
        vp.getDerivativesMV(Nderiv, imp, timeit=False)

    # Create CharPol
    # The charpol explotation is done only on root process
    if rank == 0:
        C = ee.CharPol.from_recursive_mult(extracted)
        # Locate EP3
        # Choose Npts=1 for speed-up automatic testing. Npts=2 or 3 is more realistic.
        s = C.iterative_solve((None,
                               (3.5 - 3.8j, 7+0j),
                               (3.5 - 3.8j, 7+0j)), decimals=4, algorithm='lm', Npts=1)
        delta, sol, deltaf = C.filter_spurious_solution(s, plot=False, tol=2e-2)

        # Compute error with reference solution
        D = distance_matrix(sol[:, 1][:, None], nu_ref[:, None])
        row_ind, col_ind = linear_sum_assignment(D)
        error = D[row_ind, col_ind].sum()
    else:
        # Create return variable on the other processes
        C, sol, error = None, None, None
    # Broadcast all output data
    C = comm.bcast(C, root=0)
    sol = comm.bcast(sol, root=0)
    error = comm.bcast(error, root=0)

    # Print report
    my_range = np.diff(imp.K[0].getOwnershipRange())[0]
    print(f'I am {rank} on {comm.Get_size()} processus, \
          I own {my_range} rows, global error on EP location is {error}.')
    return C, sol, error


# %% Main
if __name__ == '__main__':
    C, sol, error = main(plot=True)
    # Compare with reference solution in the complex plane
    # Because of symmetry mu and nu value are interchangeable
    # Convention differ from the papers (x1j)
    plt.plot(sol[:, 1].real, sol[:, 1].imag, 'ko', markerfacecolor='none', label='PCP')
    plt.plot(sol[:, 2].real, sol[:, 2].imag, 'ko', markerfacecolor='none', label='PCP')
    plt.plot(nu_ref.real, nu_ref.imag, 'rs', label='ref.', markerfacecolor='none', markersize='10')
    plt.plot(np.array(C.nu0).real, np.array(C.nu0).imag, 'b*', label=r'$\nu_0$',
             markerfacecolor='none', markersize='6')
    plt.legend()
    plt.xlabel(r'Re $\nu$')
    plt.ylabel(r'Im $\nu$')
