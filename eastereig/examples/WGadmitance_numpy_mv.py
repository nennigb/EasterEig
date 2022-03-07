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
Localization of exceptional point on a 2D waveguides lined with two adminttances
numpy version.
Resolution use the admintance Y=1/Z to avoid singularities with the impedance Z.

This allow to find EP3.
```
   y=h ____________________________ admintance bc (nu)

           | |
          -|-|-->     oo-duct
           | |

   y=0 ~~~~~~~~~~~~Y~~~~~~~~~~~~~~ admintance bc (mu)

```

Description:
------------
This problem is described in sec. 4.1 of arxiv.org/abs/1909.11579
and yield a **generalized eigenvalue problem** and the eigenvalue lda stands
for the axial wavenumber. The admintance is Y=1/Z.

[(k**2*Mmat + Yeff*GamMat - Kmat) - lda * Mmat ]x=0

FIXME : put good value in the example

WArning, mu and nu are multiplied by 1j wrt to Manu and Jane papers.

Examples
--------
>>> import numpy as np
>>> import eastereig as ee
>>> EP1, vp1 = main() # doctest: +ELLIPSIS
Instance of Operator class Ynumpy @nu0=(0.0012325073357595624-0.0010079264261048232j) (5 dof, height=1.0)...

Get exceptional point location and check its value
>>> EP1.EP_loc[0]
(0.001367...-0.001094...j)

"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# standard
import sys
import numpy as np
import scipy as sp
from scipy.special import factorial
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
# eastereig
import eastereig as ee
import sympy as sym
import pypolsys
import time

import matplotlib
matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
matplotlib.rcParams.update({'font.size': 12})
class Ynumpy(ee.OPmv):
    """Create a subclass of the interface class OP that describe the problem operator."""
    def __init__(self, y, n, h, rho, c, k):
        """Initialize the problem.

        Parameters
        ----------
            y : iterable
                admintance values (mu, nu)
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
        """ Define the object representation
        """
        return "Instance of Operator class {} @nu0={} ({} dof, height={})".format(self.__class__.__name__, self.nu0, self.n, self.h)

    def _mass(self):
        """ Define the mass matrix, of 1D FEM with ordered nodes.

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
        M = sp.sparse.diags([m1, m2, m1], [-1, 0, 1],
                            shape=(n, n), format='csc').toarray()
        # Boundary points
        M[0, 0] = m2/2.
        M[0, 1] = m1
        M[n-1, n-2] = m1
        M[n-1, n-1] = m2/2.

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
        # value for inner K row [k1 k2 k1]
        k1 = -1. / self.Le
        k2 = 2. / self.Le

        # Striffness matrix
        # Interior grid points
        K = sp.sparse.diags([k1, k2, k1], [-1, 0, 1], shape=(n, n), format='csc').toarray()
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

        Parameters
        ----------
        z0 : complex
            The impedance value
        """
        n = self.n
        # create Gamma matrix (complex)
        # Striffness matrix
        Gam_top = sp.sparse.coo_matrix((np.array([1.]), (np.array([0]), np.array([0]))),
                                   shape=(n, n)).toarray()
        Gam_bot = sp.sparse.coo_matrix((np.array([1.]), (np.array([n-1]), np.array([n-1]))),
                                    shape=(n, n)).toarray()
        # Store it
        self._GamMat = {'mu': Gam_bot, 'nu': Gam_top} # mu, nu

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
        """ Define the sucessive derivative of the $\tilde{K}$ matrix with respect to nu

        L = \tilde{K}- lda M (standard FEM formalism)
        with polynomial formlism L = K0 + lda K1 + ...
        thus K0=\tilde{K}

        Parameters
        ----------
        m, n : int
            the order of derivation for each variable.

        Returns
        -------
        Kn : Matrix (petsc or else)
            The n-derivative of global K0 matrix
        """
        if (m, n) == (0, 0):
            Kn = self.K[0]
        elif (m, n) == (1, 0):
            omega = self.k * self.c    # angular freq.
            Kn = self._GamMat['mu'] * 1j*self.rho*omega
        elif (m, n) == (0, 1):
            omega = self.k * self.c    # angular freq.
            Kn = self._GamMat['nu'] * 1j*self.rho*omega
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
            the order of derivation for each variable.
        Returns
        -------
        Kn : Matrix (petsc or else)
            The n-derivative of global K1 matrix
        """
        # if (m, n) = 0 return M, else 0
        if (m, n) == (0, 0):
            return -self._Mmat
        # if n!= 0 return 0 because M is constant
        else:
            return 0


def convert2polsys(p0):
    """ First attempt to export the system to polsys, based on sympy.

    N = 2
    n_coef_per_eq = np.array([6, 6], dtype=np.integer)
    all_coef = np.array([-9.80e-04, 9.78e+05,  -9.80e+00, -2.35e+02, 8.89e+04, -1.00e+00,
                         -1.00e-02, -9.84e-01, -2.97e+01,  9.87e-03,-1.24e-01, -2.50e-01], dtype=complex)
    all_deg = np.zeros((np.sum(n_coef_per_eq), N), dtype=np.integer)
    """
    # number of variables
    N = len(p0.gens)
    P = []
    # Build formal system
    for n in range(0, N):
        P.append(p0.diff((p0.gens[0], n)))
    coeff_list = []
    deg_list = []

    # Conversion
    n_coef_per_eq = np.zeros((N,), dtype=np.int64)
    for n, p in enumerate(P):
        pd = p.as_dict()
        k = 0
        for key, val in pd.items():
            coeff_list.append(val)
            deg_list.append(list(key))
            k += 1
        # store the number of terms for this polynomial
        n_coef_per_eq[n] = k

    all_coef = np.array(coeff_list, dtype=complex)
    all_deg = np.array(deg_list, dtype=np.int64)

    return N, n_coef_per_eq, all_coef, all_deg

def check_EP(lda, nu):
    """Check if a triple (lda, nu, mu) is a solution at k."""
    duct = Ynumpy(y=nu, n=N, h=h, rho=rho0, c=c0, k=k0)
    L = duct.createL(lda)
    return L.shape[0] - np.linalg.matrix_rank(L)

def pypolsys_solve(p0, degrees=None, dense=True):
    """Solve EP3 with pypolsys."""
    if degrees:
        p = drop_higher_degree(p0, degrees)
    else:
        p = p0
    out4polsys = convert2polsys(p)
    if dense:
        out4polsys = pypolsys.utils.toDense(*out4polsys)
    # with open('admitance.pkl','wb') as f:
    #     pickle.dump(out4polsys, f)
    pypolsys.polsys.init_poly(*out4polsys)
    pypolsys.polsys.init_partition(*pypolsys.utils.make_h_part(3))        
    tic = time.time()
    # The solution may be quite sensitive to the tracktol especially for low order pols
    # If Nderiv=4 1e-3- 1e-5 works well 
    # If Nderiv=3 1e-4 only works well
    #bplp = pypolsys.polsys.solve(1e-4, 1e-10, 1e-14, dense=dense)
    bplp = pypolsys.polsys.solve(1e-5, 1e-8, 0, dense=dense)
    toc = time.time()
    print('pypolsys time :', toc-tic)
    r = pypolsys.polsys.myroots.copy()
    return bplp, r

def drop_higher_degree(p, degrees):
    """Remove higher degree (tuple) for the Sympy poly object
    """
    p_dict = p.as_dict()
    for key, val in list(p_dict.items()):
        if (np.array(key) > np.array(degrees)).any():
            del p_dict[key]
    return sym.Poly.from_dict(p_dict, p.gens)


def locate_ep3(r, rt):
    """Locate EP3 as a 'common' solution between to solution set."""
    from scipy.spatial.distance import cdist
    D = cdist(r[0:3, :].T, rt[0:3, :].T, lambda u, v: np.abs(u-v).sum())
    index = np.nonzero(D < tol_locate)
    for i in range(index[0].size):
        print(r[0:3, index[0][i]])
        print(rt[0:3, index[1][i]])
        print('\n')
    return index

def direct_solve_MV(nu):
    """Direct solve with FEM model, the lda are sorted by magnitude."""
    # Create discrete operator of the pb
    imp = Ynumpy(y=nu, n=N, h=h, rho=rho0, c=c0, k=k0)
    # Initialize eigenvalue solver for *generalized eigenvalue problem*
    imp.createSolver(pb_type='gen')
    # run the eigenvalue computation
    Lambda = imp.solver.solve(nev=Nmodes, target=0+0j, skipsym=False)
    return Lambda

def distance_lda(lda, n_closest=3):
    """Compute the max distance to the n_closest neighbor."""
    x, y = lda.real, lda.imag
    points = np.column_stack([x, y])
    tree = KDTree(points)
    d_points = np.zeros((points.shape[0],))
    for n, point in enumerate(points):
        dd, ii = tree.query(point, k=n_closest)
        print(n, dd, ii)
        d_points[n] = dd.max()
    return d_points

def compute_approx_error(C, shift_modulus):
    """Compute the eigenvalue from the CharPol using a complex shift.

    Compute the error using several shift value
    TODO need to developp for the paper!
    """
    Nlda = len(C.dLambda)
    nu0 = C.nu0
    shift = np.exp(1j*0.3) * shift_modulus
    nu_shift = np.array(nu0) + shift
    # Prediction using CharPol
    an = C.eval_an_at(nu_shift)
    lda_C = np.roots(an)
    # Direct Computation
    lda_bf = direct_solve_MV(nu_shift)
    # Prediction using just a single eig approx
    lda_T = np.zeros((Nlda,), dtype=complex)
    for n, dLambda in enumerate(C.dLambda):
        Tlda = ee.charpol.Taylor(dLambda, C.nu0)
        lda_T[n] = Tlda.eval_at(nu_shift)
    # Error estimator : pick the Nlda // 2 minimal distances
    D_C = cdist(lda_bf[:Nlda, np.newaxis], lda_C[:, np.newaxis],
                lambda u, v: np.abs(u-v))
    row_ind, col_ind = linear_sum_assignment(D_C)
    err_C = np.sort(D_C[row_ind, col_ind])[-1]

    D_T = cdist(lda_bf[:Nlda, np.newaxis], lda_T[:, np.newaxis],
                lambda u, v: np.abs(u-v))
    row_ind, col_ind = linear_sum_assignment(D_T)
    err_T = np.sort(D_T[row_ind, col_ind])[-1]
    return err_C, err_T


def compute_error_wtr_number_of_modes(extracted, step=2, rhov=np.linspace(0, 10, 20)):
    """Run systematic distance check for all solution candidates in `sol`.
    """
    Nmax = len(extracted)
    Nv = np.arange(2, Nmax, step)
    ERR_C = np.zeros((rhov.size, Nv.size))
    ERR_T = np.zeros((rhov.size, Nv.size))
    for n, N in enumerate(Nv):
        CN = ee.CharPol(extracted[0:N])
        for i, rho in enumerate(rhov):
            err_C, err_T = compute_approx_error(CN, rho)
            ERR_C[i, n] = err_C
            ERR_T[i, n] = err_T
            
    # plot the Error
    plt.figure('Error')
    for n, N in enumerate(Nv):
        plt.plot(rhov, np.log10(ERR_C[:, n]), linestyle='-', marker='.', label='N='+str(Nv[n]))
    plt.legend()
    plt.xlabel(r'$|\delta|$')
    plt.ylabel(r'$\log_{10} E$')
    
    # plt.figure('Error T')
    # for n, N in enumerate(Nv):
    #     plt.plot(rhov, np.log10(ERR_T[:, n]), linestyle='-', marker='.', label='N='+str(Nv[n]))
    # plt.legend()
    # plt.xlabel(r'$|\delta|$')
    # plt.ylabel(r'$\log_{10} E$')
    return ERR_C

# %% Main
if __name__ == '__main__':
    import time
    import pickle
    import eastereig as ee
    from sympy.solvers.polysys import solve_generic
    import numpy as np
    np.set_printoptions(linewidth=150)
    """ run the example
    
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
    N = 200
    pi = np.pi
    # Number of derivative
    Nderiv = (6, 6)
    # wavenumer and air properties
    rho0, c0, k0 = 1., 1., 1.
    # duct height
    h = 1.
    # initial imepdance guess
    # nu0 = [3.0 + 3j, 3.0 + 3.0j]
    # nu0 = [0.0+0.00j, 0.]
    nu0_manu = np.array([3.1781+ 4.6751j, 3.0875 + 3.6234j])
    nu0 = np.array([1.5, 0.8])*nu0_manu/1j
    # nu0 = [1 -1.j,  0 -3.j]
    # nu0 = (0j, 0j)
    # Use to find the distance
    tol_locate = 0.1
    
    # number of dof
    # N=5 -> move to function
    
    # number of mode to compute
    Nmodes = 13

    # Create discrete operator of the pb
    imp = Ynumpy(y=nu0, n=N, h=h, rho=rho0, c=c0, k=k0)
    print(imp)
    # Initialize eigenvalue solver for *generalized eigenvalue problem*
    imp.createSolver(pb_type='gen')
    # run the eigenvalue computation
    Lambda = imp.solver.solve(nev=Nmodes, target=0+0j, skipsym=False)
    # create a list of the eigenvalue to monitor
    lda_list = np.arange(0, Nmodes)
    # return the eigenvalue and eigenvector in a list of Eig object
    extracted = imp.solver.extract(lda_list)
    # destroy solver (important for petsc/slepc)
    imp.solver.destroy()
    print('> Eigenvalue :', Lambda)
    print('> alpha/pi :', np.sqrt(k0 - Lambda)/np.pi)
    
    print('> Get eigenvalues derivatives ...\n')
    for vp in extracted:
        vp.getDerivativesMV(Nderiv, imp)

    print('Create CharPol...')
    C = ee.CharPol(extracted)
    
    # From analyical solution:
    alpha_s = 4.1969 - 2.6086j
    lda_s = k0**2 - alpha_s**2
    nu_s = np.array((3.1781+ 4.6751j, 3.0875 + 3.6234j))/ 1j
    nu_s2 = np.array((3.1781+ 4.6751j, 3.0875 + 3.6234j, 3.6598 + 7.9684j, 3.6015+6.9459j, 3.9800+11.189j, 3.9371 +10.176j, 1.0119+4.6029j, 1.0041+7.7896j))/ 1j
    
    val_s = np.array((lda_s, *nu_s))
    C.eval_at(val_s)
    
    val0 = np.array((extracted[0].lda, *nu0))
    # Locate (lda, EP)
    tic = time.time()
    sol = C.newton(((lda_s*0.9, lda_s*4.),
                    (2+2j, 4+8j),
                    (2+2j, 4+8j)), decimals=4, Npts=7)
    toc = time.time()
    print('Newton time : ', toc - tic)
    # Best stragety is 1st run coarse and second run to refine.
    # sort roots by nu_i modulus
    # expression solution in alpha
    # sol[:, 0] = np.sqrt(k0**2 - sol[:, 0])
    # s = C._newton(C.EP_system, C.jacobian, val0, verbose=True)
    
    # # sort all solution "by amplitude"
    # ind = np.argsort(np.sum(np.abs(sol), axis=1))
    # sol_ = sol[ind, :]
    # # # print them in reversed order to see the good one at the end
    # print(sol_[::-1,:])
    
        # return imp, sol
    # have a lookk on coef convergence radius
    for i, a in enumerate(C.dcoefs):
        if i>0:
            r_nu0 = np.roots(a[0, :][::-1])
            r_nu1 = np.roots(a[:, 0][::-1])
            plt.plot(r_nu0.real, r_nu0.imag, '+')
            plt.plot(r_nu1.real, r_nu1.imag, 'x')
    # if __name__=='__main__':
    #     """Show graphical outputs and reconstruction examples."""
    #     imp, sol = main()

    
    # %% play with Groebner
    p0, variables = ee.CharPol.taylor2sympyPol(C.dcoefs, tol=1e-5)
    
    # %% Compute error
    ERR_C = compute_error_wtr_number_of_modes(extracted, rhov=np.linspace(0, 10, 20))
    
    # %% pypolsys solve
    is_pypolsys = False
    dense=False
    if is_pypolsys:
        bplp, r = pypolsys_solve(p0, dense=dense)
        r[1, :] += nu0[0]
        r[2, :] += nu0[1]
        # truncate
        bplpt, rt = pypolsys_solve(p0, degrees=(p0.degree(0), p0.degree(1) -1 , p0.degree(2) - 1), dense=dense)
        rt[1, :] += nu0[0]
        rt[2, :] += nu0[1]
        # for n in range(bplp):
        #     ldan, nun, mun = r[0:3, n]
        #     print(n, ldan, nun, mun)
        plt.figure('compare')
        plt.plot(r[1, :].real, r[1,:].imag, 'bo', markerfacecolor='none', label='Homotopy '+ str((p0.degree(1), p0.degree(2))))
        # Newton may miss the important values. eg nu0= array([7.013-4.767j, 2.899-2.47j ]), Nderiv=7
        # plt.plot(sol[:, 1].real, sol[:, 1].imag, 'bo', label='nu0 (newton)', markerfacecolor='none')
        plt.plot(rt[1, :].real, rt[1,:].imag, 'k+', label='Homotopy '+ str((p0.degree(1) -1 , p0.degree(2) - 1)))
        plt.plot(nu_s.real, nu_s.imag, 'rs', label='ref.', markerfacecolor='none', markersize='10')
        plt.xlim([0, 12])
        plt.ylim([0, -5])
        plt.legend()
        plt.xlabel(r'Re $\nu$')
        plt.ylabel(r'Im $\nu$')
        
        # %% Figuer for CFA
        plt.figure('compare Manu notation')
        # invert divistion by 1j
        plt.plot(-r[1, :].imag, r[1,:].real, 'ko', markerfacecolor='none', label='Homotopy '+ str((p0.degree(1), p0.degree(2))))
        # Newton may miss the important values. eg nu0= array([7.013-4.767j, 2.899-2.47j ]), Nderiv=7
        # plt.plot(sol[:, 1].real, sol[:, 1].imag, 'bo', label='nu0 (newton)', markerfacecolor='none')
        plt.plot(-rt[1, :].imag, rt[1,:].real, 'k+', label='Homotopy '+ str((p0.degree(1) -1 , p0.degree(2) - 1)))
        plt.plot(-nu_s2.imag, nu_s2.real, 'rs', label='ref.', markerfacecolor='none', markersize='10')
        plt.plot(-np.array(nu0).imag, np.array(nu0).real, 'b*', label=r'$\boldsymbol{\nu}_0$', markerfacecolor='none', markersize='6')
        plt.xlim([0, 5])
        plt.ylim([0, 12])
        plt.legend()
        plt.xlabel(r'Re $\nu$')
        plt.ylabel(r'Im $\nu$')
        
        
        # find the closest point in both set
        index = locate_ep3(r, rt)
        
        # truncate and do the same
    # for k,v in p0.as_dict().items():
    #     print(k, v)

    groeb = False
    if groeb:
        F = [p0, p0.diff(variables[0]), p0.diff(variables[0], 2)]
        g = sym.groebner(F, method='f5b', domain='CC')

    # %% directionnal convergence
    # def radii(a):
    #     """ compute the radii of convergence.
    #     reduce to a 1D series, by putting mu = exp(1i phi)
    #     """
    #     Ndir = 10
    #     Phi = np.linspace(0, pi/2, Ndir)
    #     rho = np.zeros_like(Phi)
    #     Rho = []
    #     for phi in Phi:
    #         alp = np.tan(phi)
    #         cn = np.zeros(a.shape[1], dtype=complex) + 1e-15
    #         for n in range(0, a.shape[1]):
    #             for m in range(0, a.shape[0]):
    #                 cn[n] += a[m, n] * (alp**m)
    #         rho = 1/(np.abs(cn) ** ( - 1 / np.arange(0, a.shape[1])))
    #         Rho.append(rho)
    #     return Rho

    def radii_fit(a):
        """ LMS fit to find radii of convergence.
        """
        alp1, alp2 = np.meshgrid(np.arange(0, a.shape[0]),
                                 np.arange(0, a.shape[1]))

        alp = alp1 + alp2 + 0.001
        expo = 1 / (alp)
        # log c^0 = 0 * log c
        expo[0, 0] = 0
        # d = np.power(np.abs(a), expo)
        # print(1/d)

        V = np.hstack( (-alp1.reshape(-1, 1) / alp.reshape(-1, 1), -alp2.reshape(-1, 1)/ alp.reshape(-1,1)))
        # x, y = np.linalg.pinv(V) @ np.log(d.reshape(-1, 1))
        x, y = np.linalg.pinv(V) @ (expo.reshape(-1, 1) * np.log(np.abs(a).reshape(-1, 1)))
        return np.exp(x), np.exp(y)



    # r = radii(C.dcoefs[2])
    # Not yet finished
    if_radii = True
    if if_radii:
        for n, a in enumerate(C.dcoefs):
            if n > 0:
                x, y =radii_fit(a)
                print(x, y)

   
    # NewOption = sym.Options(g.gens, {'domain': 'CC'})
    # # sol = _solve_reduced_system2(F, F[0].gens)
    # sol_sympy = solve_generic(g, NewOption)