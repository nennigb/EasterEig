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

import numpy as np
import eastereig as ee
import matplotlib.pyplot as plt
from bruteforce import bruteForceSolvePar
import itertools as it
import concurrent.futures
from functools import partial

try:
    import sympy as sym
    sym.init_printing(forecolor='White')
except ImportError():
    print('sympy is needed for testing')


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

        # mandatory -----------------------------------------------------------
        self._lib = 'numpy'
        # create the operator matrices
        self.K = [Kmat, Mmat]
        # define the list of function to compute  the derivatives of each operator matrix
        self.dK = [self._dstiff, self._dmass]
        # define the list of function to set the eigenvalue dependance of each operator matrix
        self.flda = [None, ee.lda_func.Lda]
        # ---------------------------------------------------------------------

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

def _inner(l, N, var, nu, mu):
    """ Inner loop for sympy_check. Usefull for // computation.
    """
    dlda = np.zeros(N, dtype=np.complex)
    for n in it.product(*map(range, N)):
        print(n)
        dlda[n] = sym.N(sym.diff(l, mu, n[0], nu, n[1]).subs(var))
    return dlda

def sympy_check(nu0, sympyfile):
    """ Check multiple derivatives with sympy.
    
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
    N = (1, 1)
    # FIXME hard coded parameters
    m = 1.
    k = 1.
    mu, nu, lda = sym.symbols('mu, nu, lambda', complex=True)
    var = {mu: nu0[0], nu: nu0[1]}
    M = - sym.Matrix([[m, 0, 0],
                      [0, m, 0],
                      [0, 0, m]])
    K = sym.Matrix([[mu+k, -k, 0.],
                    [-k, 2*k, -k],
                    [0., -k, nu+k]])
    # Symbols
    p0 = sym.det(K + lda*M)
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

    np.savez(sympyfile, dlda=dlda)

    return dlda, EP_sym

def sympy_jac_check(p0, val):
    """ Validation of Jacobian matrix computation
    """

    gens = p0.gens
    lda, nu0, nu1 = gens
    N = len(val)
    P = [p0.diff((lda, i)) for i in range(0, N)]
    J = np.zeros((N, N), dtype=np.complex)
    v = np.zeros((N,), dtype=np.complex)
    for row in range(0, N):
        v[row] = np.complex(P[row].subs(dict(zip(gens, val))))
        for col in range(0, N):
            J[row, col] = np.complex(sym.diff(P[row], gens[col]).subs(dict(zip(gens, val))))
    return J, v










# %% MAIN
if __name__ == '__main__':
    from scipy import optimize
    from sympy.solvers.polysys import solve_generic
    np.set_printoptions(linewidth=150)
    Nderiv = (5, 5)
    # tuple of the inital param
    nu0 = (1., 1.) # problem si (1, 1)
    # Sympy check parameters
    sym_check = False
    sympyfile =  'sympy_dlda'

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

    for vp in extracted:
        vp.getDerivativesMV(Nderiv, net)

    C = ee.CharPol(extracted)
    p0, variables = ee.CharPol.taylor2sympyPol(C.dcoefs)
    _lda, _nu0, _nu1 =  variables
    # check eval_at method

    # Need to add a test
    # vals = (1. +1j, 0.7+1j, -1.2+1j)
    vals = (1.+1j, 0.7-1j, -1.2+0.3j)
    cp = C.eval_at(vals)
    p0_ = p0.subs({_lda:vals[0], _nu0: vals[1], _nu1: vals[2]})
    J = C.jacobian(vals)
    Jsym, vsym = sympy_jac_check(p0, vals)
    v = C.EP_system(vals)
    val_ep = np.array((2.0000000000001172-1.73205080756872j,
                       0.5000000000000471-2.598076211353326j,
                       1.499999999999996-2.5980762113533107j))
    # test NR
    sol = C.newton(((1-2j, 3+3j),
                    (-3-3j, 3+3j),
                    (-3-3j, 3+3j)), decimals=8)


    # recover the good polynomial
    # sym.roots(p0.subs({_nu0: 0, _nu1: 0}))
    # @nu0=(0.5, 0.7). Eigenvalue : [0.36708537+0.j 1.60198927+0.j 3.23092537+0.j]
    # sym.roots(p0.subs({nu0: -0.5, nu1: -0.3}))

    # sol = C.groebner_solve()
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
    # %% Check with sympy
    if sym_check:
        dlda_ref, EP_sym = sympy_check(nu0, sympyfile)
        # computed for mu=nu=1
        """
        /!\ wrong sign here
        dlda_ref[0].real
        array([[-3.41421356, -0.25      , -0.22097087, -0.1875    , -0.02900243],
               [-0.25      ,  0.13258252,  0.0625    , -0.06214806, -0.28125   ],
               [-0.22097087,  0.0625    ,  0.10358009,  0.09375   , -0.22204983],
               [-0.1875    , -0.06214806,  0.09375   ,  0.39425174,  0.5625    ],
               [-0.02900243, -0.28125   , -0.22204983,  0.5625    ,  3.47561795]])

        dlda_ref[1].real
        array([[-0.58578644, -0.25      ,  0.22097087, -0.1875    ,  0.02900243],
               [-0.25      , -0.13258252,  0.0625    ,  0.06214806, -0.28125   ],
               [ 0.22097087,  0.0625    , -0.10358009,  0.09375   ,  0.22204983],
               [-0.1875    ,  0.06214806,  0.09375   , -0.39425174,  0.5625    ],
               [ 0.02900243, -0.28125   ,  0.22204983,  0.5625    , -3.47561795]])

        dlda_ref[2].real
        array([[-2.00000000e+00, -5.00000000e-01, -8.39220777e-18,  3.75000000e-01,  7.55586683e-18],
               [-5.00000000e-01,  4.02810988e-18, -1.25000000e-01,  8.51605573e-18,  5.62500000e-01],
               [-8.39220777e-18, -1.25000000e-01,  3.71511123e-18, -1.87500000e-01,  7.77698184e-18],
               [ 3.75000000e-01,  1.80035419e-18, -1.87500000e-01,  6.16975297e-17, -1.12500000e+00],
               [ 7.55586683e-18,  5.62500000e-01, -3.48957050e-17, -1.12500000e+00, -7.23310784e-16]])
        """