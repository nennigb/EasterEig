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
##Define the OP class

To locate EP or to reconstruct the eigenvalue loci, high order derivatives of
the eigenvalue are required. These derivatives \(\lambda^(n)\) are used
in the `class EP` to create analytic functions g and h.

In practice, derivatives of the selected pair of eigenvalues can be recursively
computed by solving the bordered matrix [Andrew:1993]
$$\begin{bmatrix}
\mathbf{L} & \partial_\lambda \mathbf{L} \mathbf{x} \\
\mathbf{v}^t & 0
\end{bmatrix}
\begin{pmatrix}
\mathbf{v}^{(n)}\\ \lambda^{(n)}
\end{pmatrix}
= \begin{pmatrix}
\mathbf{F}_n\\ 0
\end{pmatrix},
$$
where the right hand side (RHS) vector \(\mathbf{F}_n \) contains terms arising
from previous order derivatives. All these terms are obtained automatically thank
to the splitting between `K`, `dK` and `flda` and the `getRHS` method during its
call by `Eig` class objects.
"""
# ee
from . import eigSolvers
from .adapter import adaptVec, adaptMat  # adapter patern to avoid interface missmatch
from . import lda_func
from .utils import multinomial_index_coefficients, multinomial_multiindex_coefficients
from eastereig import _petscHere
from abc import ABC, abstractmethod


# list of available solver
if _petscHere:
    _SOLVER_DICT = {'petsc': eigSolvers.PetscEigSolver,
                    'numpy': eigSolvers.NumpyEigSolver,
                    'scipysp': eigSolvers.ScipySpEigSolver}
else:
    _SOLVER_DICT = {'numpy': eigSolvers.NumpyEigSolver,
                    'scipysp': eigSolvers.ScipySpEigSolver}


# Abstract class, not instanciable
class OP(ABC):
    """
    The OP class define the operator of the problem.

    This is an abstract class
    and you need to subclass it to describe your own problem.

    The following attribute **must** be defined in your subclass (see the examples)

    Attributes
    ----------
    K: list
        A list of the discrete operator matrices
    dK: list
        A list of function to compute the derivative % nu of each matrix of K
    fdla: list
        A list of function that give the dependancy % lda
    """

    SOLVER_DICT = _SOLVER_DICT
    """ list of available solver. Such solver are define in eigSolver class
    """

    def __init__(self):
        """Init method."""

    def setnu0(self, nu0):
        """Set the nominal value of the parameter [mandatory]."""
        self.nu0 = nu0

    def createL(self, lda):
        """Create operator L matrix at a fixed Lambda from K and flda**(0).

        Parameter
        ---------
        lda : complex
            the eigenvalue

        Returns
        -------
        L : matrix
            operator evaluation @(lda,nu0)
        """
        # self.K=self._ImpMat() # work if we recompute K.... but why ?
        # loop over opertor matrix
        for i, Ki in enumerate(self.K):
            flda_ = self.flda[i]
            if i == 0:
                # need to initialise L, need copy ! L is a Matrix not an adapter because of copy
                # L need to be complex since lda are generally complex in non-hermitian case
                # fixme crash is Ki is real and lda complex
                if flda_ is None:
                    L = adaptMat(Ki, self._lib).copy()
                else:
                    L = adaptMat(Ki, self._lib).copy() * flda_(0, 0, [lda])  # list expected
            else:
                if flda_ is None:
                    L += Ki
                else:
                    L += Ki * flda_(0, 0, [lda])  # list expected
        # store results
        return L

    def createDL_ldax(self, vp):
        r"""Create Vector L1x for the bordered matrix.

        This vector is \(\partial_\lambda L x\) computed at nu0.

        The derivative with respect to lda is computed using using K, flda and dlda_flda

        Parameters
        ----------
        vp : Eig
            the eigenvalue object

        Returns
        -------
        L1x : vector
            \( \partial_\lambda L x \) computed at (nu0, lda)
        """
        # init L1x_
        lda = vp.lda
        x = adaptVec(vp.x, self._lib)
        L1x = adaptVec(x.duplicate(), self._lib)
        L1x.set(0.)

        # loop over opertor matrices
        for i, Ki in enumerate(self.K):
            dflda = lda_func._dlda_flda[self.flda[i]](lda)
            if dflda != 0:
                # matrix operation with the adapter return a real matrix or vector type
                L1x.obj += adaptMat(Ki, self._lib).dot(x.dot(dflda))

        # store results
        return L1x.obj

    def getRHS(self, vp, n):
        """Compute RHS vector as defined in the Andrew, Chu, Lancaster method.

        The computation might depend on nu, L, the number of derivatives and the eigenvalue
        and eigenvector derivatives. The concrete implementation of the
        RHS must be consistent with the chosen library (scipy, petsc, ...).

        Symbolic computation of the RHS for polynomial and generalized polynomial eigenvalue
        problem :

        [K_0 + f_1(lda)*K_1 + ... + + f_n(lda)*K_n ]x=0

        [K_0 + lda*K_1 + ... + + lda**n * K_n ]x=0

        The operator should be described by 3 lists
        K=[K0,K1,K2] that contains the operator matrix
        dK=[dK0,dK1,dK2] that contains the derivatives of operator matrix
        flda = [None, lin, quad] that contains the function of the eigenvalue. The function will
        return their nth derivatives for general dependancy Faa di Bruno foruma should be used.

        Parameters
        ----------
        vp : Eig
            the eigenvalue object

        Returns
        -------
        F.obj : vector (petsc,numpy,scipy)
            the RHS vector is the good format
        """
        # use adapter !
        lib = self._lib
        # adapt the class interface to be independant of the library
        x = adaptVec(vp.x, self._lib)

        # init
        F = adaptVec(x.duplicate(), lib)  # RHS same shape as eigenvector
        F.set(0.)

        # number of terms, usually 3 (K_i * f_i(lda) * xi), sometimes 2
        NTERM = 3
        # tuple to remove, because not in the RHS
        #        (Matrix, eigenvector, eigenvalue)
        # Remarks : the lda**(n) are remove in the lda_func
        skip2 = {(0, n)}     # K_0**(0) x**(n)
        skip3 = {(0, n, 0)}  # K_1**(0) x**(n) dla**(0)

        # Init matrix derivative index at previous step, to force 1st computation
        m0old = -1
        # loop over operator matrices
        for (Kid, K) in enumerate(self.K):
            # TODO caching the matrix
            # How many terms for liebnitz 2 or 3
            if self.flda[Kid] is None:
                ntermi = NTERM - 1  # 2
                skip_set = skip2
            else:
                ntermi = NTERM  # 3
                skip_set = skip3

            # multinomial index and coef
            mind, mcoef = multinomial_index_coefficients(ntermi, n)
            for (mi, m) in enumerate(mind):
                # check if index belong to RHS
                if (m not in skip_set):
                    # Computing the operator derivative may be long, the matrix is cached
                    # until its derivation order change
                    if m[0] != m0old:
                        # if matrix order derivative has changed since last computation, compute it
                        dK_m0_ = self.dK[Kid](m[0])
                    if dK_m0_ is not int(0):
                        # if matrix derivative do not vanish...
                        dK_m0 = adaptMat(dK_m0_, lib)  # FIXME may be 0
                        dx_m1 = adaptVec(vp.dx[m[1]], lib)       # usually never 0
                        # compute the eigenvalue m[2]-th derivative, return 0 if skiped
                        if ntermi == 2:
                            F.obj -= dK_m0.dot(dx_m1.dot(mcoef[mi]))
                        else:
                            # filter if lda**(n) or d_lda L lda**(n) because not in RHS
                            dlda_m = self.flda[Kid](m[2], n, vp.dlda)
                            if abs(dlda_m) != 0:
                                F.obj -= dK_m0.dot(dx_m1.dot(dlda_m*mcoef[mi]))
                m0old = m[0]

        return F.obj

    def createSolver(self, pb_type='gen', opts=None):
        """Factory function that create a eigSolver object.

        Computation are delegated to the object define in `SOLVER_DICT`.

        Parameters
        ----------
        pb_type : string
            define the type of eigenvalue problem in \n
              - 'std': K[0] X = lda X
              - 'gen' : (K[0]  + lda K[1]) X =0
              - 'PEP' : (K[0] + K[1] +lda**2 K[2] ) X= 0
        opts : not used now

        Returns
        -------
        solver : EigSolver object
            the computation should be realize with the solver interface
        """
        lib = self._lib
        # create solver
        self.solver = OP.SOLVER_DICT[lib](self.K, pb_type)


# Abstract class, not instanciable
class OPmv(OP):
    """Absract class to manage multivariate problems."""

    def getRHS(self, vp, n):
        """Compute RHS vector as defined in the Andrew, Chu, Lancaster method.

        The computation might depend on nu, L, the number of derivatives and the eigenvalue
        and eigenvector derivatives. The concrete implementation of the
        RHS must be consistent with the chosen library (scipy, petsc, ...).


        Symbolic computation of the RHS for polynomial and generalized polynomial eigenvalue
        problem :

        [K_0 + f_1(lda)*K_1 + ... + + f_n(lda)*K_n ]x=0

        [K_0 + lda*K_1 + ... + + lda**n * K_n ]x=0

        The operator should be described by 3 lists
        K=[K0,K1,K2] that contains the operator matrix
        dK=[dK0,dK1,dK2] that contains the derivatives of operator matrix
        flda = [None, lin, quad] that contains the function of the eigenvalue. The function will
        return their nth derivatives for general dependancy Faa di Bruno foruma should be used.

        Parameters
        ----------
        vp : Eig
            the eigenvalue object

        Returns
        -------
        F.obj : vector (petsc,numpy,scipy)
            the RHS vector is the good format
        """
        # use adapter !
        lib = self._lib
        # adapt the class interface to be independant of the library
        x = adaptVec(vp.x, self._lib)

        # init
        F = adaptVec(x.duplicate(), lib)  # RHS same shape as eigenvector
        F.set(0.)

        # number of terms, usually 3 (K_i * f_i(lda) * xi), sometimes 2
        NTERM = 3
        # get number of variable involved in the derivatives
        nvar = len(n)
        # tuple to remove, because not in the RHS
        #        (Matrix, eigenvector, eigenvalue)
        # Remarks : the lda**(n) are remove in the lda_func
        # skip2 = {(0, n)}     # K_0**(0) x**(n)
        skip2 = {tuple([(0,)*nvar, n])}     # K_0**(0, ..., 0) x**(n1, ..., n_nvar)
        # skip3 = {(0, n, 0)}  # K_1**(0) x**(n) dla**(0)
        skip3 = {tuple([(0,)*nvar, n, (0,)*nvar])}  # K_1**(0, ..., 0) x**(n1, ...,n_nvar) dla**(0, ..., 0)

        # Init matrix derivative index at previous step, to force 1st computation
        m0old = -1
        # loop over operator matrices
        for (Kid, _) in enumerate(self.K):
            # TODO caching the matrix
            # How many terms for liebnitz 2 or 3
            if self.flda[Kid] is None:
                ntermi = NTERM - 1  # 2
                skip_set = skip2
            else:
                ntermi = NTERM  # 3
                skip_set = skip3

            # multinomial index and coef
            mind, mcoef = multinomial_multiindex_coefficients(ntermi, n)
            for (mi, m) in enumerate(mind):
                # check if index belong to RHS
                if tuple(m) not in skip_set:
                    # Computing the operator derivative may be long, the matrix is cached
                    # until its derivation order change
                    if m[0] != m0old:
                        # if matrix order derivative has changed since last computation, compute it
                        dK_m0_ = self.dK[Kid](*m[0])
                    if dK_m0_ is not int(0):
                        # if matrix derivative do not vanish...
                        dK_m0 = adaptMat(dK_m0_, lib)  # FIXME may be 0
                        dx_m1 = adaptVec(vp.dx[m[1]], lib)       # usually never 0
                        # compute the eigenvalue m[2]-th derivative, return 0 if skiped
                        if ntermi == 2:
                            F.obj -= dK_m0.dot(dx_m1.dot(mcoef[mi]))
                        else:
                            # filter if lda**(n) or d_lda L lda**(n) because not in RHS
                            dlda_m = self.flda[Kid](m[2], n, vp.dlda)
                            if abs(dlda_m) != 0:
                                F.obj -= dK_m0.dot(dx_m1.dot(dlda_m*mcoef[mi]))
                m0old = m[0]
        del m0old, dK_m0_, dK_m0, dx_m1
        return F.obj
