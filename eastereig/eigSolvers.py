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

""" ##Definiftion of the eigenvalue solver class for numpy, scipy.sparse and petsc/slepc solvers

This module implement standard, generalized and quadratic eigenvalue problems.

Examples
--------
The basic use of this class is

1. Create the solver object :
 `myOP.createSolver(lib='numpy', pb_type='gen')`
2. Solve :
 `myOP.solver.solve(nev=6, target=0+0j, skipsym=False)`
3. Get back the eigenvalue and eigenvector in a list of Eig object :
 `extracted = myOP.solver.extract([0, 1])`
4. Destroy solver (usefull for petsc/slepc) :
 `myOP.solver.destroy()`

Remarks
-------
The petsc version is configured to use shift and invert with mumps...
"""

from abc import ABC, abstractmethod
import scipy as sp
from scipy.sparse.linalg import eigs
import numpy as np

from eastereig import _petscHere, Eig, gopts

if _petscHere:
    from slepc4py import SLEPc
    from petsc4py import PETSc
    Print = PETSc.Sys.Print


# compatible with Python 2 *and* 3:
# ABC = ABCMeta('ABC', (object,), {'__slots__': ()})

class EigSolver(ABC):
    """Define the abstract interface common to all Eigenvalue solver.

    Attributes
    ----------
    K : matrix or matrix list
        contains the eigenvalue problem matrices
    pb_type: string {'std','gen','PEP''}
        the type of eigenvalue problem to solve
    Lda : list
        List of the sorted eigenvalues
    """

    def __init__(self, K, pb_type):
        """Init the solver with pb_type and the list of matrix K."""
        # store pb_type
        self.pb_type = pb_type
        # create a link to the eigenvalue problem matrices (from parent object)
        self.K = K

    def extract(self, eig_list):
        """Extract the eig_list eigenvectors and return a type of Eig object.

        Parameters
        ----------
        eig_list : iterable
            index list (related to the sort criteria) of the wanted eigenvalue

        Returns
        -------
        extrated : list
            list of Eig objects associated to eig_list
        """
        extracted = []
        # loop over the modes
        for i in eig_list:
            extracted.append(Eig(self._lib, self.Lda[i], self.Vec[:, i]))

        return extracted

    def destroy(self):
        """Destroy the solver."""
        pass

    def sort(self, key='abs', skipsym=False):
        """Sort the eigenvalue problem by order of magnitude.

        Important to be sure that all solver use the same criteria

        Parameters
        ----------
        key : 'abs' (default), 'real', 'imag'
            the key used to sort the eigenavlues
        skipsym : bool
            remove eigenvalue with imag lda< 0

        Remarks
        ------
        collective
        """
        key_func_dict = {'abs': np.abs, 'imag': np.imag, 'real': np.real}
        if key in key_func_dict.keys():
            key_func = key_func_dict[key]
        else:
            raise AttributeError('This type of key is not defined')

        if skipsym:
            # Filter left going waveguide modes
            skip_idx = np.where(self.Lda.imag > 0)[0]  # unpack tuple
            self.Lda = self.Lda[skip_idx]

        # sort by assending order of key
        self.idx = np.argsort(key_func(self.Lda))  # like in matlab
        self.Lda = self.Lda[self.idx]
        # if symmetric filtering is active need global index for eigenvector
        if skipsym:
            self.idx = skip_idx[self.idx]

        self.nconv = len(self.Lda)

    @abstractmethod
    def solve(self, nev=6, target=0+0j, key='abs', skipsym=False):
        """Solve the eigenvalue problem."""
        pass


class NumpyEigSolver(EigSolver):
    """Define the concrete interface common to all numpy Eigenvalue solver.

    The eigenvalue problem is solved with numpy for full matrix
    """

    # keep trace of the lib
    _lib = 'numpy'

    def solve(self, nev=6, target=0+0j, skipsym=False):
        """Solve the eigenvalue problem and get back the results as (Lda, X).

        Parameters
        ----------
        nev : int
            number of requested eigenpairs
        target : complex, optional
            target used for the shift and invert transform
        skipsym : bool
            remove eigenvalue with imag lda< 0

        Remarks
        --------
        For full matrix, all eigenvalues are obtained. Neither 'nev' nor 'target' are used. These parameters
        are used to ensure a common interface between solvers.
        """
        print('> Solve {} eigenvalue problem with {} class...\n'.format(self.pb_type,
                                                                        self.__class__.__name__))
        # FIXME need to modify order based on lambda func
        if self.pb_type == 'std':
            self.Lda, Vec = sp.linalg.eig(self.K[0], b=None)
        elif self.pb_type == 'gen':
            self.Lda, Vec = sp.linalg.eig(self.K[0], b=-self.K[1])
        elif self.pb_type == 'PEP':
            self.Lda, Vec = self._pep(self.K)
        else:
            raise NotImplementedError('The pb_type {} is not yet implemented'.format(self.pb_type))

        # Sort eigenvectors and create idx index
        self.sort(skipsym=skipsym)
        self.Vec = Vec[:, self.idx]
        return self.Lda

    @staticmethod
    def _pep(K):
        """Polynomial eigenvalue solver by linearisation with numpy.

        The linearization is performed with the first companion form (as in slepc).
        Because of the monomial form, it is recommended to not exceed a degree of 5.
        For higher degree, using orthogonal polynomial basis is recommanded.

        Parameters
        ----------
        K : List
            list of matrix. The order is (K[0] + K[1]*lda + K[2]*lda**2)x=0

        Examples
        --------
        This example comes from polyeig function in matlab documentation
        >>> M = np.diag([3, 1, 3, 1])
        >>> C = np.array([[0.4, 0, -0.3, 0], [0, 0, 0, 0],[-0.3, 0, 0.5, -0.2],[0, 0, -0.2, 0.2]])
        >>> K = np.array([[-7, 2, 4, 0], [2, -4, 2, 0], [4, 2, -9, 3], [ 0, 0, 3, -3]])
        >>> lda_ref = np.array([ -2.449849443705628e+00, -2.153616198037310e+00, -1.624778340529248e+00, \
                                 2.227908732047911e+00, 2.036350976643702e+00, 1.475241143475665e+00, \
                                 3.352944297785435e-01, -3.465512996736311e-01])
        >>> x1_ref =  np.array([  -1.827502943493723e-01,    -3.529667428394975e-01,     5.360280532739473e-01,    -7.447756269292356e-01])
        >>> D,X = NumpyEigSolver._pep([K, C, M])
        >>> np.linalg.norm(D-lda_ref)<1e-12
        True
        >>> x1 = X[:,0]/np.linalg.norm(X[:,0])
        >>> np.linalg.norm(x1-x1_ref)<1e-12
        True
        """
        shape = K[0].shape
        dtype = K[0].dtype
        degree = len(K) - 1
        degree1 = degree - 1
        # Create auxiliary matrix
        I = np.eye(*shape, dtype=dtype)
        Z = np.zeros(shape, dtype=dtype)
        # Create companion matrix
        Comp = np.empty((degree, degree), dtype=object)
        for (i, j), _ in np.ndenumerate(Comp):
            if i == degree1:
                # last list line with K
                Comp[i, j] = -K[j]
            elif j == i + 1:
                # fill 1-st diag
                Comp[i, j] = I
            else:
                # zeros elsewhere
                Comp[i, j] = Z

        # Fill with I on the main diagonal excepted last term
        Diag = np.empty((degree, degree), dtype=object)
        for (i, j), _ in np.ndenumerate(Diag):
            if (i == j) and (i < degree1):
                Diag[i, j] = I
            elif (i == j) and (i == degree1):
                Diag[i, j] = K[j+1]
            else:
                # zeros elsewhere
                Diag[i, j] = Z

        A = np.bmat(Comp.tolist())
        B = np.bmat(Diag.tolist())
        # solved linearised QEP
        D, V = sp.linalg.eig(A, B)
        # the (2*N,) eigenvector are normalized to 1.
        V = V[0:shape[0], :]
        return D, V


class ScipySpEigSolver(EigSolver):
    """Define the concrete interface common to all numpy Eigenvalue solver.

    The eigenvalue problem is solved with numpy for full matrix
    """

    # keep trace of the lib
    _lib = 'scipysp'

    def solve(self, nev=6, target=0+0j, skipsym=False):
        """Solve the eigenvalue problem and get back the results as (Lda, X).

        Parameters
        ----------
        nev : int
            number of requested eigenpairs
        target : complex, optional
            target used for the shift and invert transform
        skipsym : bool
            remove eigenvalue with imag lda< 0

        Remarks
        --------
        For full matrix all eigenvalues are obtained. nev is not used.
        """
        print('> Solve eigenvalue {} problem with {} class...\n'.format(self.pb_type,
                                                                        self.__class__.__name__))
        if self.pb_type == 'std':
            self.Lda, Vec = eigs(self.K[0], k=nev, M=None, sigma=target, return_eigenvectors=True)
        elif self.pb_type == 'gen':
            self.Lda, Vec = eigs(self.K[0], k=nev, M=-self.K[1],
                                 sigma=target, return_eigenvectors=True)
        elif self.pb_type == 'PEP':
            self.Lda, Vec = self._pep(self.K, k=nev, sigma=target)
        else:
            raise NotImplementedError('The pb_type {} is not yet implemented'.format(self.pb_type))

        # Sort eigenvectors and create idx index
        self.sort(skipsym=skipsym)
        self.Vec = Vec[:, self.idx]
        return self.Lda

    @staticmethod
    def _pep(K, k=4, sigma=0.):
        """Polynomial eigenvalue solver by linearisation with scipy sparse.

        The linearization is performed with the first companion form (as in slepc).
        Because of the monomial form, it is recommended to not exceed a degree of 5.
        For higher degree, using orthogonal polynomial basis is recommanded.

        Parameters
        ----------
        K: List
            list of matrix. the order is (K[0] + K[1]*lda + K[2]*lda**2)x=0
        k: Int
            The number of requested eigenpairs.
        sigma: complex
            The value arround which eigenvalues are looked for.

        Examples
        --------
        This example comes from polyeig function in matlab documentation
        >>> M = sp.sparse.csc_matrix(np.diag([3, 1, 3, 1]), dtype=complex)
        >>> C = sp.sparse.csc_matrix(np.array([[0.4, 0, -0.3, 0], [0, 0, 0, 0],[-0.3, 0, 0.5, -0.2],[0, 0, -0.2, 0.2]], dtype=complex))
        >>> K = sp.sparse.csc_matrix(np.array([[-7, 2, 4, 0], [2, -4, 2, 0], [4, 2, -9, 3], [ 0, 0, 3, -3]], dtype=complex))
        >>> lda_ref = np.array([ -2.153616198037310e+00, -1.624778340529248e+00,  2.036350976643702e+00, 1.475241143475665e+00, \
                                 3.352944297785435e-01, -3.465512996736311e-01])
        >>> x1_ref =  np.array([-3.421456419701390e-01, 9.295577535525387e-01, 4.558756372896405e-02, -1.295396330257916e-01])
        >>> D,X = ScipySpEigSolver._pep([K, C, M], k=6)
        >>> ind=np.argsort(D)  # normally -2.15 is the first
        >>> np.linalg.norm(D[ind] - np.sort(lda_ref))<1e-12
        True
        >>> x1 = X[:,ind[0]]/np.linalg.norm(X[:,ind[0]]) # assume X[:,0] <-> lda=-2.15
        >>> np.linalg.norm(np.abs(x1)- np.abs(x1_ref))<1e-12
        True
        """
        shape = K[0].shape
        dtype = K[0].dtype
        degree = len(K) - 1
        degree1 = degree - 1
        # create auxiliary matrix
        I = sp.sparse.eye(*shape, dtype=dtype).tocsc()
        Z = None

        # Create companion matrix with 'None'
        Comp = np.empty((degree, degree), dtype=object)
        for (i, j), _ in np.ndenumerate(Comp):
            if i == degree1:
                # last list line with K
                Comp[i, j] = -K[j]
            elif j == i + 1:
                # fill 1-st diag
                Comp[i, j] = I

        # Fill with I on the main diagonal excepted last term
        Diag = np.empty((degree, degree), dtype=object)
        for (i, j), _ in np.ndenumerate(Diag):
            if (i == j) and (i < degree1):
                Diag[i, j] = I
            elif (i == j) and (i == degree1):
                Diag[i, j] = K[j+1]

        # FIXME see the impact of .tocsc
        A = sp.sparse.bmat(Comp.tolist()).tocsc()
        B = sp.sparse.bmat(Diag.tolist()).tocsc()

        # solved linearised QEP
        D, V = eigs(A, k=k, M=B, sigma=sigma, return_eigenvectors=True)
        # the (2*N,) eigenvector are normalized to 1.
        V = V[0:shape[0], :]
        return D, V


# Define only if petsc and slepc are present
if _petscHere:
    class PetscEigSolver(EigSolver):
        """Define the concrete interface common to all PETSc/SLEPc Eigenvalue solver.

        Configured to use shift and invert transform with mumps
        """

        # TODO if move into fonction, no need to add a test if poetscHere ?
        PB_TYPE_FACTORY = {
            'std': SLEPc.EPS,
            'gen': SLEPc.EPS,
            'PEP': SLEPc.PEP
        }
        """ list of petsc factory depending on the kind of eigenvalue problem
        """

        PB_TYPE = {
            'std': SLEPc.EPS.ProblemType.NHEP,
            'gen': SLEPc.EPS.ProblemType.GNHEP,
            'PEP': SLEPc.PEP.ProblemType.GENERAL
        }
        """
        SLEPc problem type dictionnary, by defaut use only *non-hermitian*
        """

        # keep trace of the lib
        _lib = 'petsc'

        def __init__(self, K, pb_type):
            """Init slepc with the good `pb_type`."""
            # check input
            if pb_type not in self.PB_TYPE_FACTORY.keys():
                raise NotImplementedError('The pb_type {} is not yet implemented...'.format(self.pb_type))
            # store pb type
            self.pb_type = pb_type
            self._SLEPc_PB = PetscEigSolver.PB_TYPE_FACTORY[pb_type]
            # create a link to the eigenvalue problem matrices (from parent object)
            self.K = K

        def _create(self, nev, target):
            """Create and setup the SLEPC solver."""
            # mpi stuff
            comm = PETSc.COMM_WORLD.tompi4py()
            rank = comm.Get_rank()
            # create the solver with the selected factory
            E = self._SLEPc_PB()
            E.create()

            # Setup the spectral transformation
            SHIFT = SLEPc.ST().create()
            SHIFT.setType(SHIFT.Type.SINVERT)
            E.setST(SHIFT)
            E.setTarget(target)
            # operator setup
            K = self.K
            if self.pb_type == 'std':
                # unpack the operator matrix
                E.setOperators(*K)
            if self.pb_type == 'gen':
                # M=-K1
                E.setOperators(K[0], -K[1])
            else:
                E.setOperators(K)

            # number of eigenvlue we are looking for
            E.setDimensions(nev=nev)
            # By defaut use non hermitian solver
            E.setProblemType(self.PB_TYPE[self.pb_type])

            # Direct solver seting (for shift and invert)
            ksp = SHIFT.getKSP()
            ksp.setType('preonly')  # direct solver in petsc= preconditioner
            pc = ksp.getPC()
            pc.setType('lu')
            # pc.setFactorSolverType('superlu_dist')

            # set solver options
            pc.setFactorSolverType(gopts['direct_solver_name'])
            opts = PETSc.Options(gopts['direct_solver_petsc_options_name'])
            for op_name, op_value in gopts['direct_solver_petsc_options_dict'].items():
                opts[op_name] = op_value
            # Mumps options to avoid mumps crash with high fill in
            # The problem arise if the prediction/mem allocation is too different (default 20%)
            # opts["icntl_14"] = 50 # ICNTL(14) controls the percentage increase in the estimated working space
            # opts["icntl_6"] = 5  # Use ICNTL(6)= 5 for augmented system (which is asystem with a large zero diagonal block).

            #  After all options have been set, the user should call the setFromOptions()
            E.setFromOptions()

            # store the solver
            self.E = E

        def destroy(self):
            """Destroy the petsc/slecp solver."""
            self.E.destroy()

        def extract(self, eig_list):
            """Extract the eig_list eigenvectors.

            Parameters
            ----------
            eig_list : iterable
                index list (related to the sort criteria) of the wanted eigenvalue

            Returns
            -------
            extrated : list
                list of Eig objects associated to eig_list
            """
            # init output
            extracted = []
            # create petsc vector
            vr = self.K[0].createVecRight()
            # loop over the modes
            for i in eig_list:
                lda = self.E.getEigenpair(self.idx[i], vr)
                extracted.append(Eig(self._lib, lda, vr.copy()))

            return extracted

        def solve(self, nev=6, target=0+0j, key='abs', skipsym=False):
            """Solve the eigenvalue problem and get back the results as (Lda, X).

            Parameters
            ----------
            nev : int
                number of requested eigenpairs
            target : complex, optional
                target used for the shift and invert transform
            skipsym : bool
                remove eigenvalue with imag lda< 0

            Remarks
            --------
            It is still possible to interact with the SLEPc solver until the destroy method call
            """
            self._create(nev, target)
            Print('> Solve eigenvalue {} problem with {} class ...\n'.format(self.pb_type,
                                                                             self.__class__.__name__))
            self.E.solve()

            nconv = self.E.getConverged()
            self.Lda = np.zeros((nconv,), dtype=np.complex128)

            # extract *unsorted* eigenvalue
            for i in range(0, nconv):
                self.Lda[i] = self.E.getEigenpair(i)  # collective
            # sort
            self.sort(key=key, skipsym=skipsym)

            return self.Lda
