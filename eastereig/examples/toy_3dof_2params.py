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
depending on the two complex parameters mu and nu. We would like to find the EP3.

Examples
--------
>>> C, sol, error = main() # doctest: +ELLIPSIS
> Solve gen eigenvalue problem with NumpyEigSolver class...
>>> error < 1e-3
True

@author: bn
"""
import numpy as np
import eastereig as ee
import itertools as it
import concurrent.futures
from functools import partial
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import sympy as sym
from sympy.solvers.polysys import solve_generic


class Network(ee.OPmv):
    """Create subclass of OPmv that describe the multiparameter problem."""

    def __init__(self, nu0):
        """Initialize the problem.

        Parameters
        ----------
            nu0 : tuple
                The parameters initial Values (complex).
        """
        self.m = 1
        self.k = 1
        self.nu0 = nu0

        # assemble *base* matrices
        Mmat = self._mass()
        Kmat = self._stiff()

        # initialize OP interface
        self.setnu0(nu0)

        # mandatory -----------------------------------------------------------
        self._lib = 'numpy'
        # create the operator matrices
        self.K = [Kmat, Mmat]
        # define the list of function to compute  the derivatives of each operator matrix
        self.dK = [self._dstiff, self._dmass]
        # define the list of function to set the eigenvalue dependance of each operator matrix
        self.flda = [None, ee.lda_func.Lda]
        # ---------------------------------------------------------------------

    def __repr__(self):
        """Define the object representation."""
        text = "Instance of Operator class {} @nu0={}."
        return text.format(self.__class__.__name__, self.nu0)

    def _mass(self):
        """Define the a diagonal mass matrix."""
        m = self.m
        M = - np.array([[m, 0, 0],
                       [0, m, 0],
                       [0, 0, m]], dtype=complex)
        return M

    def _stiff(self):
        """Define the stifness matrix of 1D FEM with ordered nodes."""
        mu = self.nu0[0]
        nu = self.nu0[1]
        k = self.k
        K = np.array([[mu+k, -k, 0.],
                      [-k, 2*k, -k],
                      [0., -k, nu+k]], dtype=complex)
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
        mu, nu = self.nu0

        if (m, n) == (0, 0):
            Kn = self.K[0]
        elif (m, n) == (1, 0):
            Kn = np.zeros_like(self.K[0], dtype=complex)
            Kn[0, 0] = 1.
        elif (m, n) == (0, 1):
            Kn = np.zeros_like(self.K[0], dtype=complex)
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


def _inner(l, N, var, nu, mu):
    """Inner loop for sympy_check. Usefull for // computation."""
    dlda = np.zeros(N, dtype=complex)
    for n in it.product(*map(range, N)):
        print(n)
        dlda[n] = sym.N(sym.diff(l, mu, n[0], nu, n[1]).subs(var))
    return dlda


def sympy_check_poly():
    """Compute the true characteristic polynomial with sympy."""
    # FIXME hard coded parameters
    m = 1.
    k = 1.
    mu, nu, lda = sym.symbols('mu, nu, lambda', complex=True)

    M = - sym.Matrix([[m, 0, 0],
                      [0, m, 0],
                      [0, 0, m]])
    K = sym.Matrix([[mu+k, -k, 0.],
                    [-k, 2*k, -k],
                    [0., -k, nu+k]])
    # Symbols
    p0 = sym.det(K + lda*M)
    # Ensure the polynomial has the good sign
    a3 = p0.coeff(lda, 3)
    p0 = p0 / a3
    return p0, mu, nu, lda


def sympy_check(nu0, sympyfile=None):
    """Check multiple derivatives with sympy.

    Parameters
    ----------
    n0 : tuple
        Contains the nominal value of mu and nu.
    sympyfile : string
        filename for saving sympy output.

    Returns
    -------
    dlda : list
        Contains the sucessive derivative with respect to mu (row) and
        nu (column) as np.array

    """
    # set max // workers
    max_workers = 3
    # derivatives order
    N = (2, 2)
    p0, mu, nu, lda = sympy_check_poly()
    var = {mu: nu0[0], nu: nu0[1]}
    p1 = sym.diff(p0, lda)
    p2 = sym.diff(p1, lda)

    # solve with groebner
    F = [p0, p1, p2]
    g = sym.groebner(F, method='f5b', domain='CC')
    NewOption = sym.Options(g.gens, {'domain': 'CC'})
    # sol = _solve_reduced_system2(F, F[0].gens)
    EP_sym = solve_generic(g, NewOption)

    # solve for lda
    ldas = sym.solve(p0, lda)
    dlda = dict()
    print('May take a while...')

    # use partial to fixed all function parameters except lda
    _inner_lda = partial(_inner, N=N, var=var, nu=nu, mu=mu)
    # run the // computation
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, dlda_ in enumerate(executor.map(_inner_lda, ldas)):
            dlda[i] = dlda_
    if sympyfile:
        np.savez(sympyfile, dlda=dlda)
    return dlda, EP_sym, g, p0


def sympy_jac_check(p0, s):
    """Check the Jacobian matrix computation.

    Parameters
    ----------
    p0: sympy expression
        The true characteristic polynomial.
    s : iterable
        Absolute values of the extened paramters (lda, nu).
    """
    gens = tuple(p0.free_symbols)
    gens = sorted(gens, key=lambda x: x.name, reverse=False)
    lda, _, _ = gens
    N = len(s)
    s = np.array(s, dtype=complex)
    # s[1::] = s[1::] - np.array(nu0, dtype=complex)
    P = [p0.diff((lda, i)) for i in range(0, N)]
    J = np.zeros((N, N), dtype=complex)
    v = np.zeros((N,), dtype=complex)
    for row in range(0, N):
        v[row] = complex(P[row].subs(dict(zip(gens, s))))
        for col in range(0, N):
            J[row, col] = complex(sym.diff(P[row], gens[col]).subs(dict(zip(gens, s))))
    return J, v


sol_ana = np.array([[2. - 1.77635684e-15j, 1. + 1.41421356e+00j, 1. - 1.41421356e+00j],
                    [2. + 1.77635684e-15j, 1. - 1.41421356e+00j, 1. + 1.41421356e+00j],
                    [2. - 1.73205081e+00j, 0.5-2.59807621e+00j, 1.5-2.59807621e+00j],
                    [2. + 1.73205081e+00j, 0.5+2.59807621e+00j, 1.5+2.59807621e+00j],
                    [2. - 1.73205081e+00j, 1.5-2.59807621e+00j, 0.5-2.59807621e+00j],
                    [2. + 1.73205081e+00j, 1.5+2.59807621e+00j, 0.5+2.59807621e+00j]])


# Ref jacobian matrix obtained from sympy_jac_check with s = (1.5, 1., 2.)
Jana = np.array([[-0.25+0.j,  0.25+0.j,  0.75+0.j],
                 [-5. + 0.j,  2. + 0.j,  1. + 0.j],
                 [6. + 0.j, -2. + 0.j, -2. + 0.j]])

# Ref Ep system obtained from sympy_jac_check with s = (1.5, 1., 2.)
vana = np.array([1.625+0.j, - 0.25 + 0.j, - 5. + 0.j])


def error_between(sol1, sol2):
    """Evalute the error between two sets of unordered vectors.

    Parameters
    ----------
    sol1 : np.array
        Array wih m1 lines of n-dimentional solution.
    sol1 : np.array
        Array wih m2 lines of n-dimentional solution.

    Returns
    -------
    error : float
        The global error between the two set using the best permutation.
    """
    D = distance_matrix(sol1, sol2)
    row_ind, col_ind = linear_sum_assignment(D)
    error = D[row_ind, col_ind].sum()
    return error


def main(plot=False):
    """Find the EP3 of the 3 DOF problem.

    Return
    ------
    sol : np.array
        Fhe EP3 solutions. For each solution, it constains the eigenvalue, mu and nu.
    error : float
        The global error between the found EP3 and analytic solution.
    """
    # Define the order of derivation for each parameter
    Nderiv = (5, 5)
    # Define the inital value of parameter
    nu0 = np.array([0.95590969 - 1.48135044j + 0.15 + 0.1j,
                    1. + 1.41421356e+00j + 0.1 + 0.1j])
    # Instiate the problem
    net = Network(nu0)
    net.createSolver(pb_type='gen')
    # Run the eigenvalue computation
    Lambda = net.solver.solve(target=0+0j)
    # Create a list of the eigenvalue to monitor
    lda_list = np.arange(0, 3)
    # Return the eigenvalue and eigenvector in a list of Eig object
    extracted = net.solver.extract(lda_list)
    # destroy solver (important for petsc/slepc)
    net.solver.destroy()
    # Compute the eigenvalue derivatives
    for vp in extracted:
        vp.getDerivativesMV(Nderiv, net)
    # Find EP with Charpol
    C = ee.CharPol(extracted)
    # Use homotopy solver (if installed)
    # bplp, s = C.homotopy_solve(tracktol=1e-12, finaltol=1e-13, tol_filter=-1)
    s = C.iterative_solve((None,
                          (nu0 - (1+1j), nu0 + (1+1j)),
                          (nu0 - (1+1j), nu0 + (1+1j))), Npts=2, algorithm='lm', max_workers=4, tol=1e-5)
    delta, sol, deltaf = C.filter_spurious_solution(s, plot=plot, tol=1e-3)
    return C, sol, error_between(sol, sol_ana)
    # Export a sympy poly object


# %% MAIN
if __name__ == '__main__':
    C, sol, error = main(plot=True)
    print('Found EP3:')
    print(sol)
    print('Error between found EP and analytic solution: ', error)
    # How to generate numercal value from sympy computation
    # dlda, EP_sym, g, p0 = sympy_check(C.nu0, sympyfile=None)
    # s = (1.5, 1., 2.)
    # Jana, vana = sympy_jac_check(p0, s)
    # Jpcp = C.jacobian(s)
