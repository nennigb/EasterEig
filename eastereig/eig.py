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
##Define the Eig class and subclass

This class interact with `OP` class to compute the sucessive derivatives of the
eigenvalue and the eigenvectors.

This class defines also several approximation of the eigenvalues based on these
high order derivatives:

  - Taylor series
  - Padé approximant
  - Puiseux series (with `EP` class)
  - Analytic auxiliary functions (with `EP` class)
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
import time
import itertools as it

from eastereig import _petscHere, gopts, _CONST
from eastereig.utils import pade, Taylor

if _petscHere:
    from slepc4py import SLEPc
    from petsc4py import PETSc
    from mpi4py import MPI  # TODO find a workaround
    Print = PETSc.Sys.Print
    from ._petsc_utils import matrow, PETScVec2PETScMat


def is_sequence(seq):
    """Test if `seq` is a sequence.

    Remarks
    -------
    Based on deprecated module from `numpy.distutils.misc_util.py`.
    """
    if isinstance(seq, str):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True


def Eig(lib, *args, **kwargs):
    """ Factory function for creating the *Eig objects
    """
    EIG_FACTORY = {'petsc': PetscEig, 'numpy': NumpyEig, 'scipysp': ScipyspEig}
    """ Define the target for factory the Eig Class depending of the linear algebra lib
    """
    try:
        return EIG_FACTORY[lib](lib, *args, **kwargs)
    except KeyError():
        print("'lib' argument should be in {}".format(EIG_FACTORY.keys()))


# compatible with Python 2 *and* 3:
# ABC = ABCMeta('ABC', (object,), {'__slots__': ()})
class AbstractEig(ABC):
    """Abstract class that manage an eigenpair (lda, x) of an OP object and its derivatives.

    The derivatives are computed thanks to Andrew, Chu, Lancaster method (1994).

    Attributes
    -----------
    lib: string  {'numpy','scipysp','petsc'}
        The name of the matrix library
    lda: complex
        The eigenvalue
    x: complex array_like of type lib
        The eigenvector
    dx: list (if computed)
        The list of the sucessive derivatives of x % nu
    dlda: list (if computed)
        The list of the sucessive derivatives of lda % nu
    nu0: None, scalar or iterable
        The value is generally set whe the derivative are computed or loaded.
    """

    def __init__(self, lib, lda=None, x=None):
        """Init the instance
        """
        self._lib = lib
        self.lda = lda
        self.x = x  # must add normalisation here
        self.nu0 = None

        # init derivative if note None
        if (lda is not None) & (x is not None):
            self.dlda = [lda]
            self.dx = [x]
        else:
            self.dlda = []
            self.dx = []

    def __repr__(self):
        """ Define the representation of the class
        """
        # Check if dlda has a 'shape'  (works with np.array)
        # else use 'len' which works with all iterables...
        if hasattr(self.dlda, 'shape'):
            nd = tuple(np.array(self.dlda.shape) - 1)
        else:
            nd = len(self.dlda) - 1

        return "Instance of {}  @lda={} with #{} derivatives".format(self.__class__.__name__,
                                                                     self.lda,
                                                                     nd)

    @abstractmethod
    def export(self, filename, eigenvec=True):
        """ Export the eigenvalue and the eigenvector derivatives (if eigenvect=True) into a file

        The exported format depend on the matrix format lib.

        Parameters
        -----------
        filename : string
            the name of the exported file (without extension)
        eigenvec : bool
            export or not the eigenvector
        The created object depend on the matrix format lib.
        """
        pass

    @abstractmethod
    def load(self, filename, eigenvec=True):
        """ Load an Eig object saved in a file

        Parameters
        -----------
        filename : string
            the name of the exported file (without extension)
        eigenvec : bool
            import or not the eigenvector
        The created object depend on the matrix format lib.
        FIXME more elegant with @classmethod ?
        """
        pass

    def addD(self, dlda, dx):
        """  add new derivatives
        """
        self.dlda.extend(dlda)
        if dx is not None:
            self.dx.extend(dx)

    @abstractmethod
    def getDerivatives(self, N, L, timeit=False):
        """Compute the successive derivative of an eigenvalue of an OP instance.

        Parameters
        ----------
        N: int
            The number derivative to compute.
        L: OP
            The operator OP instance that describe the eigenvalue problem.
        timeit : bool, optional
            If `True` it activates textual profiling outputs. Default is `False`.

        RHS derivative must start at n=1 for 1st derivatives
        """
        pass

    @abstractmethod
    def getDerivativesMV(self, N, L, timeit=False):
        """Compute the successive derivatives of an eigenvalue of a multivariate OPmv instance.

        Parameters
        ----------
        N: int
            The number derivative to compute.
        L: OP
            The operator OP instance that describe the eigenvalue problem.
        timeit : bool, optional
            If `True` it activates textual profiling outputs. Default is `False`.

        RHS derivative must start at n=1 for 1st derivatives
        """
        pass

    def to_Taylor(self):
        """Convert the eigenvalue derivatives to Taylor class."""
        return Taylor.from_derivatives(np.array(self.dlda, dtype=complex),
                                       self.nu0)

    def taylor(self, points, n=-1):
        """
        Evaluate the Taylor expansion of order n at `points`.

        Parameters
        ----------
        points : array_like
            The value where the series is evaluated. Give the absolute value,
            not the relative % nu0.
            In the multivariate case, the function is not vectorized. Just put
            the computation point.
        n : int
            The number of terms considered in the expansion
            if no value is given or if n is `None`, the full array `dlda` is considered.
            if n is negative truncation are used.

        Returns
        -------
        tay : array_like
            the eigenvalue Taylor series
        """
        # Check order vs length
        if len(self.dlda) == 0:
            raise IndexError('Run getDerivative* before...\n')

        # Check if scalar
        if not is_sequence(self.nu0):
            if n is None: n = len(self.dlda)
            # Converting to np.array
            dlda = np.array(self.dlda, dtype=complex)
            # Get Taylor coef in ascending order
            Df = dlda[0:n] / sp.special.factorial(np.arange(n))
            # Polyval require higher degree first
            tay = np.polyval(Df[::-1], points - self.nu0)
        else:
            N = self.dlda.ndim
            # Create slices accounting for truncation
            slices = (slice(0, n),) * N
            print(self.dlda[slices].shape)
            T = Taylor.from_derivatives(np.array(self.dlda[slices], dtype=complex),
                                        self.nu0)
            tay = T.eval_at(points)
        return tay

    def pade(self, points, n=-1):
        """
        Evaluate the Padé expansion of order [n//2,n//2] at `points`.

        Parameters
        ----------
        points : array_like
            the value where the series is evaluated. Give the absolute value,
            not the relative % nu0.
        n : int
            The number of terms considered in the expansion
            if no value is given or if n=-1, the size of the array dlda is considered

        Returns
        -------
        pad : array_like
            the value of padé approximant at point

        Remarks
        -------
        For now, works only for scalar parameters nu.
        """
        # Check order vs length
        if len(self.dlda) == 1:
            raise IndexError('Run getDerivative* before...\n')

        if is_sequence(self.nu0):
            raise NotImplementedError(('Padé approximant works now only for scalar parameter.',
                                      'Not for nu0={}'.format(self.nu0)))

        if n == -1:
            n = len(self.dlda)
        # converting to np.array
        dlda = np.array(self.dlda, dtype=complex)
        # get Taylor coef in ascending order
        Df = dlda[0:n] / sp.special.factorial(np.arange(n))
        # order d(0) -> d(n) for padé
        p, q = pade(Df, n//2)
        pad = p(points-self.nu0) / q(points-self.nu0)
        return pad

    def puiseux(self, ep, points, index=0, n=-1):
        """
        Evaluate the Puiseux expansion with n terms at `points` .

        Parameters
        ----------
        EP : EP instance
            The EP instance that store Puiseux coefficients.
        points : array_like
            the points where the series must be computed, give the absolute value,
            not the relative % nu0.
        index : int
            the index of the EP is there multiple EP shared between both eigenvalues.
        n : integer
          The number of terms considered in the expansion
          if no value is given or if n=-1, the size of the array dlda is considered.

        Returns
        -------
        f1,f2 : array_like
            the Puiseux series evaluation of both eigenvalue
        """
        # if points.dtype
        # points_ = points.astype(complex)
        if is_sequence(self.nu0):
            raise NotImplementedError(('Puiseux series works now only for scalar parameter.',
                                       'Not for nu0={}'.format(self.nu0)))
        try:
            ep.a
        except:
            print('warning need to compute Puiseux coef before at `index`...\n')
            ep.getPuiseux(index=index)

        if n > -1:
            a1 = ep.a[index][:n]
        else:
            n = len(ep.a[index])
            a1 = ep.a[index]

        EP_loc = ep.EP_loc[index]
        a2 = a1.copy()

        f1 = np.ones(points.shape, dtype=complex)*a1[0]
        f2 = np.ones(points.shape, dtype=complex)*a1[0]
        for k in range(1, n):
            # reconstruct the 2 solutions
            f1 += a1[k]*np.power(points-EP_loc, k/2.)
            f2 += a1[k]*np.power(points-EP_loc, k/2.)*(-1)**(k)

        return (f1, f2)


    def anaAuxFunc(self, ep, points, n=-1):
        r"""
        Evalaute the analytic auxiliary functions reconstruction, based on g and h
        Taylor expansion at `points`.


        From the definitions of \(g\) and \(h\) the two eigenvalues can be written as
        $$
        \begin{align}
        \lambda_{+} &= \frac{g+\sqrt{h}}{2}, \\
        \lambda_{-} &= \frac{g-\sqrt{h}}{2}.
        \end{align}
        $$

        These expressions can be approximated through the truncated Taylor series
        \( T_h\) and \(T_g\)
        $$
        \begin{align}
        {A}_{\lambda_{+}} &= \frac{T_g+\sqrt{T_h}}{2}, \\
        {A}_{\lambda_{-}} &= \frac{T_g-\sqrt{T_h}}{2}.
        \end{align}
        $$

        Parameters
        ----------
        ep: EP instance
            The EP instance that store h and g successive derivatives
        points: array-like
            the value where the serie is evaluated. Give the absolute value, not the
            relative % nu0.
        n: integer [optional]
            The number of terms considered in the expansion
            if no value is given or if n=-1, the size of the array dlda is considered.

        Returns
        -------
        ldap,ldam : array_like
            the reconstructed eigenvalue pair
        """

        # compute the analytic auxiliary functions g and h from lda dérivatives
        ep._dh()
        ep._dg()
        if n > -1:
            # truncate
            dgTay = ep._dgTay[:n]
            dhTay = ep._dhTay[:n]
        else:
            dgTay = ep._dgTay
            dhTay = ep._dhTay

        # evaluate Taylor series
        mapg = np.polyval(dgTay[::-1], points - self.nu0)
        maph = np.polyval(dhTay[::-1], points - self.nu0)
        ldap = 0.5*(mapg+np.sqrt(maph))
        ldam = 0.5*(mapg-np.sqrt(maph))
        return ldap, ldam

    @staticmethod
    def _spellcheck():
        """Fast spell check.
        """
        print(_CONST.decode('ascii'))


class PetscEig(AbstractEig):
    """ Manage an eigenpair (lda,x)  of an OP object and its derivatives

    Concrete class for petsc matrix
    """

    def export(self, filename, eigenvec=True):
        """ Export the eigenvalue and the eigenvector derivatives (if eigenvect=True) into a file

        The exported format depend on the matrix format lib.

        Parameters
        -----------
        filename : string
            the name of the exported file (without extension)
        eigenvec : bool
            export or not the eigenvector
        The created object depend on the matrix format lib.
        """
        # export eigenvalue derivatives
        if MPI.COMM_WORLD.Get_rank() == 0:
            filename_dlda = filename + '_dlda'
            np.savez(filename_dlda, dlda=self.dlda, nu0=self.nu0, lib=self._lib)

        # export eigenvector derivatives
        if eigenvec:
            # If the file name ends with .gz it is automatically compressed when closed.
            filename_dx = filename + '_dx.petsc.gz'
            # Save *.petsc file
            ViewStd = PETSc.Viewer()    # Init. Viewer
            ViewStd.createBinary(filename_dx, mode=PETSc.Viewer.Mode.WRITE)
            for x in self.dx:
                ViewStd.view(x)         # Put PETSc object into the viewer
            ViewStd.destroy()           # Destroy Viewer

    def load(self, filename, eigenvec=True):
        """ Load an Eig object saved in a file

        Parameters
        -----------
        filename : string
            the name of the exported file (without extension)
        eigenvec : bool
            import or not the eigenvector
        The created object depend on the matrix format lib.
        FIXME more elegant with @classmethod
        """
        # export eigenvalue derivatives
        filename_dlda = filename + '_dlda.npz'
        f = np.load(filename_dlda)
        Nderiv = len(f['dlda'])

        # export eigenvector derivatives
        if eigenvec:
            # if the file name ends with .gz it is automatically compressed when closed.
            filename_dx = filename + '_dx.petsc.gz'
            ViewOpen = PETSc.Viewer()     # Init. Viewer
            ViewOpen.createBinary(filename_dx, mode=PETSc.Viewer.Mode.READ, comm=PETSc.COMM_WORLD)
            DX = []
            for i in range(0, Nderiv):
                # attention hdf5 syntax different for loading!
                w = PETSc.Vec().load(ViewOpen)
                DX.append(w)
                # Print(DX[i].view(PETSc.Viewer.STDOUT()))
            ViewOpen.destroy()

            # put in the object
            # check if extend is better
            self.addD(f['dlda'], DX)
            self.x = DX[0]
        else:
            self.addD(f['dlda'], None)
            self.x = None
        # other attributes
        self.nu0 = complex(f['nu0'])
        self.lda = f['dlda'][0]
        self._lib = f['lib']

    # TODO move parameters to config file
    @staticmethod
    def _InitDirectSolver(A, name='mumps'):
        """ Initialization of the petsc direct solver for LU factorization

        In petsc, direct solver belong to the precondition objects ksp

        Parameters
        -----------
        A : petsc Matrix
        name : petsc direct solver name(default=mumps)
            the list of available solvers depend on your petsc install {mumps, superlu_dist,
            superlu, umfpack}
        """
        # create linear solver
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)

        # Initialize ksp solver.
        ksp.setOperators(A)
        ksp.setType('preonly')
        # choose LU factorization
        pc = ksp.getPC()
        pc.setType('lu')
        # choose the solver
        pc.setFactorSolverType(name)

        #TODO Add option elsewhere
        # set up PETSc environment to read solver parameters
        # opts = PETSc.Options("mat_mumps_")
        # opts["icntl_1"] = 1 #CNTL(1) is the output stream for error messages (default 6)
        # opts["icntl_2"] = 1 #ICNTL(2) is the output stream for diagnostic printing, statistics, and warning messages. (default 0)
        # opts["icntl_3"] = 1 #ICNTL(3): output stream for global information, collected on the host (default 6 )

        return ksp

    def getDerivatives(self, N, op, timeit=False):
        """Compute the successive derivative of an eigenvalue of an OP instance.

        Parameters
        -----------
        N: int
            the number derivative to compute
        L: OP
            the operator OP instance that describe the eigenvalue problem
        timeit : bool, optional
            A flag to activate textual profiling outputs. Default is `False`.


        RHS derivative must start at n=1 for 1st derivatives
        """
        # create communicator for mpi command
        comm = PETSc.COMM_WORLD.tompi4py()

        # get nu0 value where the derivative are computed
        self.nu0 = op.nu0
        # construction de la matrice de l'opérateur L
        L = op.createL(self.lda)
        # normalization condition (push elsewhere : différente méthode, indépendace vs type )
        # must be done before L1x
        v = L.createVecRight()
        v.set(1.+0j)
        # see also VecScale
        # self.x = self.x / z.dot(x)
        self.x.scale(1/v.tDot(self.x))
        self.dx[0] = self.x

        # constrution du vecteur (\partial_\lambda L)x, ie L.L1x
        L1x = op.createDL_ldax(self)

        # bordered
        # ---------------------------------------------------------------------
        # même matrice à factoriser pour toutes les dérivées
        # Create the Nested Matrix (For now 3.9.0 petsc bug convert do not work with in_place)

        # convert z as a Matrix object
#        vmat = PETScVec2PETScMat(v) # /!\ SEEM NOT MEMORY SAFE /!\
#        #zmatT=PETSc.Mat().createTranspose(zmat) # not inplace for rect mat, not optimal
#        vmatT=PETSc.Mat()
#        vmat.transpose(vmatT)
        vmat = matrow((1, L.size[1]), np.complex128(1.))

        # PETSC Mat * Vec  return Vec
        # convert L1x as a Matrix object
        L1xmat = PETScVec2PETScMat(L1x)  # columnwize

        Bord = PETSc.Mat()
        #  C = temp matrix, ie for now 3.9.0 petsc bug convert do not work with in_place

        C = PETSc.Mat().createNest([[L, L1xmat], [vmat, None]])
        # get back the value, assume the order is the same in nest and in the converted
        ind = C.getNestISs()                                            # create IS
        # conversion from nested to aij (for mumps)
        C.convert(PETSc.Mat.Type.AIJ, Bord)

        # initialisation du solveur
        ksp = self._InitDirectSolver(Bord, name=gopts['direct_solver_name'])  # defaut mumps, non symetric...
        u = Bord.createVecLeft()

        # getSubVector :
        # This function may return a subvector without making a copy, therefore it
        # is not safe to use the original vector while modifying the subvector.
        # Other non-overlapping subvectors can still be obtained from X using this function.

        # if N > 1 loop for higher order terms
        Print('> Linear solve...')
        Zero = PETSc.Vec().create()
        Zero.setSizes(size=(None, 1))  # free for local, global size=1
        Zero.setUp()
        Zero.setValue(0, 0+0j)
        # n start now at 1 for uniformization
        for n in range(1, N+1):
            # compute RHS
            tic = time.time()  # init timer
            Ftemp = op.getRHS(self, n)
            if timeit:
                Print("              # getRHS real time :", time.time()-tic)

            Fnest = PETSc.Vec().createNest([Ftemp, Zero])
            # monolithique (no copy)
            # getArray Returns a pointer to a contiguous array that contains this processor's
            # portion of the vector data
            F = PETSc.Vec().createWithArray(Fnest.getArray())  # don't forget () !

            # n=0 get LU and solve, then solve with stored LU
            # solution u contains [dx, dlda])
            tic = time.time()  # init timer
            # F.view(PETSc.Viewer.STDOUT())
            ksp.solve(F, u)
            if timeit:
                Print("              # solve LU real time :", time.time()-tic)
            # store results as list
            # Print('indice :', ind[0][1].getIndices(),u[ind[0][1].getIndices()] )
            # self.dlda.append( np.asscalar( u[ind[0][1].getIndices()] ) )     # get value from IS, pb car //
            # get value from IS
            derivee = u[ind[0][1].getIndices()]

            if len(derivee) == 0:
                derivee = np.array([0.], dtype=np.complex128)
            # send the non empty value to all process
            derivee = comm.allreduce(derivee, MPI.SUM)
            # get lda^(n)
            self.dlda.append(derivee[0])
            self.dx.append(PETSc.Vec().createWithArray(u.getSubVector(ind[0][0]).copy()))  # get pointer from IS, need copy
            if timeit:
                Print(n)
        if timeit:
            Print('\n')

    def getDerivativesMV(self, N, op, timeit=False):
        """Compute the successive derivatives of an eigenvalue of a multivariate OPmv instance.

        Parameters
        -----------
        N: int
            The number derivative to compute.
        L: OPmv
            The operator OP instance that describe the eigenvalue problem.
        timeit : bool, optional
            If `True` it activates textual profiling outputs. Default is `False`.

        RHS derivative must start at n=1 for 1st derivatives
        """
        # create communicator for mpi command
        comm = PETSc.COMM_WORLD.tompi4py()

        # get nu0 value where the derivative are computed
        self.nu0 = op.nu0
        # Create an empty array of object
        self.dx = np.empty(N, dtype=object)
        # Create an zeros array for dlda
        self.dlda = np.zeros(N, dtype=complex)
        self.dlda.flat[0] = self.lda

        # construction de la matrice de l'opérateur L
        L = op.createL(self.lda)
        # normalization condition (push elsewhere : différente méthode, indépendace vs type )
        # must be done before L1x
        v = L.createVecRight()
        v.set(1. + 0j)
        # see also VecScale
        # self.x = self.x / z.dot(x)
        self.x.scale(1/v.tDot(self.x))
        self.dx.flat[0] = self.x

        # constrution du vecteur (\partial_\lambda L)x, ie L.L1x
        L1x = op.createDL_ldax(self)

        # bordered
        # ---------------------------------------------------------------------
        # même matrice à factoriser pour toutes les dérivées
        # Create the Nested Matrix (For now 3.9.0 petsc bug convert do not work with in_place)

        # convert z as a Matrix object
#        vmat = PETScVec2PETScMat(v) # /!\ SEEM NOT MEMORY SAFE /!\
#        #zmatT=PETSc.Mat().createTranspose(zmat) # not inplace for rect mat, not optimal
#        vmatT=PETSc.Mat()
#        vmat.transpose(vmatT)
        vmat = matrow((1, L.size[1]), np.complex128(1.))

        # PETSC Mat * Vec  return Vec
        # convert L1x as a Matrix object
        L1xmat = PETScVec2PETScMat(L1x)  # columnwize

        Bord = PETSc.Mat()
        #  C = temp matrix, ie for now 3.9.0 petsc bug convert do not work with in_place

        C = PETSc.Mat().createNest([[L, L1xmat], [vmat, None]])
        # get back the value, assume the order is the same in nest and in the converted
        ind = C.getNestISs()                                            # create IS
        # conversion from nested to aij (for mumps)
        C.convert(PETSc.Mat.Type.AIJ, Bord)

        # initialisation du solveur
        ksp = self._InitDirectSolver(Bord, name=gopts['direct_solver_name'])  # defaut mumps, non symetric...
        u = Bord.createVecLeft()

        # getSubVector :
        # This function may return a subvector without making a copy, therefore it
        # is not safe to use the original vector while modifying the subvector.
        # Other non-overlapping subvectors can still be obtained from X using this function.

        # if N > 1 loop for higher order terms
        Print('> Linear solve...')
        Zero = PETSc.Vec().create()
        Zero.setSizes(size=(None, 1))  # free for local, global size=1
        Zero.setUp()
        Zero.setValue(0, 0 + 0j)
        # n start now at 1 for uniformization
        for n in it.product(*map(range, N)):
            # Except for (0, ..., 0)
            if n != (0,)*len(N):
                # compute RHS
                tic = time.time()  # init timer
                Ftemp = op.getRHS(self, n)
                PETSc.garbage_cleanup(comm)
                if timeit:
                    Print("              # getRHS real time :", time.time()-tic)

                Fnest = PETSc.Vec().createNest([Ftemp, Zero])
                # monolithique (no copy)
                # getArray Returns a pointer to a contiguous array that contains this processor's
                # portion of the vector data
                F = PETSc.Vec().createWithArray(Fnest.getArray())  # don't forget () !

                # n=0 get LU and solve, then solve with stored LU
                # solution u contains [dx, dlda])
                tic = time.time()  # init timer
                # F.view(PETSc.Viewer.STDOUT())
                ksp.solve(F, u)
                if timeit:
                    Print("              # solve LU real time :", time.time()-tic)
                # store results as list
                # Print('indice :', ind[0][1].getIndices(),u[ind[0][1].getIndices()] )
                # self.dlda.append( np.asscalar( u[ind[0][1].getIndices()] ) )     # get value from IS, pb car //
                # get value from IS
                derivee = u[ind[0][1].getIndices()]

                if len(derivee) == 0:
                    derivee = np.array([0.], dtype=np.complex64)
                # send the non empty value to all process
                derivee = comm.allreduce(derivee, MPI.SUM)
                # get lda^(n)
                self.dlda[n] = derivee[0]
                self.dx[n] = PETSc.Vec().createWithArray(u.getSubVector(ind[0][0]).copy())  # get pointer from IS, need copy
                if timeit:
                    Print(n)
            if timeit:
                Print('\n')

# end class PetscEig


class NumpyEig(AbstractEig):
    """ Manage an eigenpair (lda,x)  of an OP object and its derivatives

    Concrete class for numpy array
    """

    def export(self, filename, eigenvec=True):
        """ Export the eigenvalue and the eigenvector derivatives (if eigenvect=True) into a file

        The exported format depend on the matrix format lib.

        Parameters
        ----------
        filename : string
            the name of the exported file (without extension)
        eigenvec : bool
            export or not the eigenvector
        The created object depend on the matrix format lib.
        """
        # export eigenvalue derivatives
        dic = {'dlda': self.dlda, 'nu0': self.nu0, 'lib': self._lib}
        if eigenvec:
            # export eigenvector derivatives
            dic.update({'dx': self.dx})
        np.savez(filename, **dic)


    def load(self, filename, eigenvec=True):
        """ Load an Eig object saved in a file

        Parameters
        -----------
        filename : string
            the name of the file to load with extension
        eigenvec : bool
            import or not the eigenvector
        The created object depend on the matrix format lib.
        FIXME more elegant with @classmethod
        """

        f = np.load(filename)
        Nderiv = len(f['dlda'])

        if eigenvec:
            dx = list(f['dx'])
            x = dx[0]
        else:
            dx, x = None, None
        # add attribute
        self.addD(f['dlda'], dx)
        self.nu0 = np.complex128(f['nu0'])
        self.lda = f['dlda'][0]
        self.x = x
        self._lib = f['lib']

    def getDerivatives(self, N, op, timeit=False):
        """Compute the successive derivative of an eigenvalue of an OP instance.

        Parameters
        ----------
        N: int
            The number derivative to compute.
        op: OP
            The operator OP instance that describe the eigenvalue problem.
        timeit: bool
            Unused for this class.

        RHS derivative must start at n=1 for 1st derivatives
        """
        # get nu0 value where the derivative are computed
        self.nu0 = op.nu0
        # construction de la matrice de l'opérateur L, ie L.L
        L = op.createL(self.lda)
        # normalization condition (push elsewhere : différente méthode, indépendace vs type )
        # must be done before L1x
        v = np.ones(shape=self.x.shape)
        # see also VecScale
        self.x *= (1/v.dot(self.x))
        self.dx[0] = self.x

        # constrution du vecteur (\partial_\lambda L)x, ie L.L1x
        L1x = op.createDL_ldax(self)

        # bordered
        # ---------------------------------------------------------------------
        # Same matrix to factorize for all RHS
        Zer = np.zeros(shape=(1, 1), dtype=complex)
        Zerv = np.zeros(shape=(1,), dtype=complex)
        Bord = np.bmat([[L, L1x.reshape(-1, 1)],
                        [v.reshape(1, -1), Zer]])  # reshape is to avoid (n,) in bmat

        # if N > 1 loop for higher order terms
        print('> Linear solve...')
        # n start now at 1 for uniformization
        for n in range(1, N+1):
            # compute RHS
            tic = time.time()  # init timer
            Ftemp = op.getRHS(self, n)
            # F= sp.bmat([Ftemp, Zerv]).reshape(-1,1)
            F = np.concatenate((Ftemp, Zerv))
            # print("              # getRHS real time :", time.time()-tic)

            tic = time.time()  # init timer
            if n == 1:
                # compute the lu factor
                lu, piv = sp.linalg.lu_factor(Bord, check_finite=False)
            # Forward and back substitution, u contains [dx, dlda])
            u = sp.linalg.lu_solve((lu, piv), F, check_finite=False)
            # print("              # solve LU real time :", time.time()-tic)

            # get lda^(n)
            derivee = u[-1]
            # store the value
            self.dlda.append(derivee)
            self.dx.append(u.copy()[:-1])
            # print(n, ' ')

    def getDerivativesMV(self, N, op, timeit=False):
        """Compute the successive derivatives of an eigenvalue of a multivariate OPmv instance.

        Parameters
        ----------
        N: tuple of int
            the number derivative to compute
        op: OPmv
            the operator OP instance that describe the eigenvalue problem
        timeit: bool
            Unused for this class.

        RHS derivative must start at n=1 for 1st derivatives
        """
        # get nu0 value where the derivative are computed
        self.nu0 = op.nu0
        # construction de la matrice de l'opérateur L, ie L.L
        L = op.createL(self.lda)
        # normalization condition (push elsewhere : différente méthode, indépendace vs type )
        # must be done before L1x
        v = np.ones(shape=self.x.shape)
        # see also VecScale
        scale = (1/v.dot(self.x))
        if np.abs(scale) < 1e6:  # TODO this tol is arbitrary
            self.x *= scale
        else:
            print('Warning : v is nearly co-linear to x (|scale|={}). Use random vector for v.'.format(abs(scale)))
            # Test (possibily) several random vector
            while np.abs(scale) > 1e2:  # TODO this tol is arbitrary
                v = np.random.rand(*self.x.shape)
                scale = (1/v.dot(self.x))
                print('          new scale is {}'.format(abs(scale)))
            self.x *= scale

        # Create an empty array of object
        self.dx = np.empty(N, dtype=object)
        self.dx.flat[0] = self.x
        # Create an zeros array for dlda
        self.dlda = np.zeros(N, dtype=complex)
        self.dlda.flat[0] = self.lda

        # constrution du vecteur (\partial_\lambda L)x, ie L.L1x
        L1x = op.createDL_ldax(self)

        # bordered
        # ---------------------------------------------------------------------
        # Same matrix to factorize for all RHS
        Zer = np.zeros(shape=(1, 1), dtype=complex)
        Zerv = np.zeros(shape=(1,), dtype=complex)
        Bord = np.bmat([[L               , L1x.reshape(-1, 1)],
                        [v.reshape(1, -1), Zer]])  # reshape is to avoid (n,) in bmat

        # if N > 1 loop for higher order terms
        print('> Linear solve...')
        # n is the deriviative multi-index (tuple)
        for n in it.product(*map(range, N)):
            # Except for (0, ..., 0)
            if n != (0,)*len(N):
                # compute RHS
                tic = time.time()  # init timer
                Ftemp = op.getRHS(self, n)
                # F= sp.bmat([Ftemp, Zerv]).reshape(-1,1)
                F = np.concatenate((Ftemp, Zerv))
                # print("              # getRHS real time :", time.time()-tic)

                tic = time.time()  # init timer
                if sum(n) == 1:
                    # compute the lu factor
                    lu, piv = sp.linalg.lu_factor(Bord)
                # Forward and back substitution, u contains [dx, dlda])
                u = sp.linalg.lu_solve((lu, piv), F)
                # print("              # solve LU real time :", time.time()-tic)

                # get lda^(n)
                derivee = u[-1]
                # store the value
                self.dlda[n] = derivee
                self.dx[n] = u.copy()[:-1]
        # print(self.dlda)

# end class NumpyEig


class ScipyspEig(AbstractEig):
    """ Manage an eigenpair (lda,x)  of an OP object and its derivatives

    Concrete class for scipy (sp)arse array
    """

    def export(self, filename, eigenvec=True):
        """ Export the eigenvalue and the eigenvector derivatives (if eigenvect=True) into a file

        The exported format depend on the matrix format lib.

        Parameters
        -----------
        filename : string
            the name of the exported file (without extension)
        eigenvec : bool
            export or not the eigenvector
        """
        # same as numpy because eigs returns full vector
        # export eigenvalue derivatives
        dic = {'dlda': self.dlda, 'nu0': self.nu0, 'lib': self._lib}
        if eigenvec:
            # export eigenvector derivatives
            dic.update({'dx': self.dx})
        np.savez(filename, **dic)

    def load(self, filename, eigenvec=True):
        """ Load an Eig object saved in a file

        Parameters
        -----------
        filename : string
            the name of the file to load with extension
        eigenvec : bool
            import or not the eigenvector
        The created object depend on the matrix format lib.
        FIXME more elegant with @classmethod
        """
        # same as numpy because eigs returns full vector

        f = np.load(filename)
        Nderiv = len(f['dlda'])

        if eigenvec:
            dx = list(f['dx'])
            x = dx[0]
        else:
            dx, x = None, None
        # add attribute
        self.addD(f['dlda'], dx)
        self.nu0 = np.complex128(f['nu0'])
        self.lda = f['dlda'][0]
        self.x = x
        self._lib = f['lib']

    def getDerivatives(self, N, op, timeit=False):
        """ Compute the successive derivative of an eigenvalue of an OP instance

        Parameters
        -----------
        N: int
            The number derivative to compute.
        L: OP
            The operator OP instance that describe the eigenvalue problem.
        timeit : bool, optional
            If `True` it activates textual profiling outputs. Default is `False`.

        RHS derivative must start at n=1 for 1st derivatives
        """

        # get nu0 value where the derivative are computed
        self.nu0 = op.nu0
        # construction de la matrice de l'opérateur L, ie L.L
        L = op.createL(self.lda)
        # normalization condition (push elsewhere : différente méthode, indépendace vs type )
        # must be done before L1x
        v = np.ones(shape=self.x.shape)
        # see also VecScale
        self.x *= (1/v.dot(self.x))
        self.dx[0] = self.x

        # constrution du vecteur (\partial_\lambda L)x, ie L.L1x
        L1x = op.createDL_ldax(self)  # FIXME change, now with return
        Zerv = np.zeros(shape=(1,), dtype=complex)
        # bordered
        # ---------------------------------------------------------------------
        # Same matrix to factorize for all RHS, conversion to scr for scipy speed
        Bord = sp.sparse.bmat([[L               , L1x.reshape(-1, 1)],
                               [v.reshape(1, -1), None]]).tocsc()  # reshape is to avoid (n,) in bmat

        # if N > 1 loop for higher order terms
        print('> Linear solve...')
        # n start now at 1 for uniformization
        for n in range(1, N+1):
            # compute RHS
            tic = time.time()  # init timer
            Ftemp = op.getRHS(self, n)
            # F= sp.bmat([Ftemp, Zerv]).reshape(-1,1)
            F = np.concatenate((Ftemp, Zerv))
            if timeit:
                print("              # getRHS real time :", time.time()-tic)

            tic = time.time()  # init timer
            if n == 1:
                # umfpack is not included in scipy but can be used with scikit-umfpack.
                # umfpack is the default choice when available. If not, scipy uses superlu
                # sp.sparse.linalg.use_solver(useUmfpack=True)
                # compute the lu factor
                lusolve = sp.sparse.linalg.factorized(Bord)

            # Forward and back substitution, u contains [dx, dlda])
            u = lusolve(F)
            if timeit:
                print("              # solve LU real time :", time.time()-tic)

            # get lda^(n)
            derivee = u[-1]
            # store the value
            self.dlda.append(derivee)
            self.dx.append(u.copy()[:-1])
            print(n, ' ')

        print('\n')

    def getDerivativesMV(self, N, op, timeit=False):
        """Compute the successive derivatives of an eigenvalue of a multivariate OPmv instance.

        Parameters
        ----------
        N: tuple of int
            the number derivative to compute
        op: OPmv
            the operator OP instance that describe the eigenvalue problem
        timeit: bool
            If `True` it activates textual profiling outputs. Default is `False`.

        RHS derivative must start at n=1 for 1st derivatives
        """
        # get nu0 value where the derivative are computed
        self.nu0 = op.nu0
        # construction de la matrice de l'opérateur L, ie L.L
        L = op.createL(self.lda)
        # normalization condition (push elsewhere : différente méthode, indépendace vs type )
        # must be done before L1x
        v = np.ones(shape=self.x.shape)
        # see also VecScale
        self.x *= (1/v.dot(self.x))
        # Create an empty array of object
        self.dx = np.empty(N, dtype=object)
        self.dx.flat[0] = self.x
        # Create an zeros array for dlda
        self.dlda = np.zeros(N, dtype=complex)
        self.dlda.flat[0] = self.lda

        # constrution du vecteur (\partial_\lambda L)x, ie L.L1x
        L1x = op.createDL_ldax(self)  # FIXME change, now with return
        Zerv = np.zeros(shape=(1,), dtype=complex)
        # bordered
        # ---------------------------------------------------------------------
        # Same matrix to factorize for all RHS, conversion to scr for scipy speed
        Bord = sp.sparse.bmat([[L, L1x.reshape(-1, 1)],
                               [v.reshape(1, -1), None]]).tocsc()  # reshape is to avoid (n,) in bmat

        # if N > 1 loop for higher order terms
        print('> Linear solve...')
        # n start now at 1 for uniformization
        for n in it.product(*map(range, N)):
            # Except for (0, ..., 0)
            if n != (0,)*len(N):
                # compute RHS
                tic = time.time()  # init timer
                Ftemp = op.getRHS(self, n)
                # F= sp.bmat([Ftemp, Zerv]).reshape(-1,1)
                F = np.concatenate((Ftemp, Zerv))
                if timeit:
                    print("              # getRHS real time :", time.time()-tic)

                tic = time.time()  # init timer
                if sum(n) == 1:
                    # umfpack is not included in scipy but can be used with scikit-umfpack.
                    # umfpack is the default choice when available. If not, scipy uses superlu
                    # sp.sparse.linalg.use_solver(useUmfpack=True)
                    # compute the lu factor
                    lusolve = sp.sparse.linalg.factorized(Bord)

                # Forward and back substitution, u contains [dx, dlda])
                u = lusolve(F)
                if timeit:
                    print("              # solve LU real time :", time.time()-tic)

                # get lda^(n)
                derivee = u[-1]
                # store the value
                self.dlda[n] = derivee
                self.dx[n] = u.copy()[:-1]
                print(n, ' ')

# end class ScipyspEig
