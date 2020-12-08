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
# Define CharPol class and auxilliary classes.

This class with handle the **partial characteristic polynomial**.
This polynomial is built from Vieta formulas and from the successive
derivatives of the selected set of eigenvalues.
[Vieta' formulas](https://en.wikipedia.org/wiki/Vieta%27s_formulas)

"""


import numpy as np
from numpy.linalg import norm
import itertools as it
from eastereig.eig import AbstractEig
from eastereig.utils import _outer, diffprodTree, div_factorial, diffprodMV
# from eastereig import EP
# TODO is still sympy usefull
import sympy as sym

# // evaluation in newton method
import concurrent.futures
from functools import partial

# multidimentional polyval
# /!\ numpy.polynomial.polynomial.polyval != np.polyval
# from numpy.polynomial.polynomial import polyval
# import numpy.polynomial.polyutils as pu
from eastereig.fpoly.fpoly import polyvalnd



class CharPol():
    r""" Numerical representation of the *partial* characteristic polynomial.

        This polynomial is built from Vieta formulas and from the successive
        derivatives of the selected set of eigenvalues \(\Lambda\).

        This class allows to reconstruct eigenvalue loci and to locate
        EP between the eigenvalues from the set.

        This polynomial can be seen as a polynomial in \(\lambda\), whom coefficients \(a_n\\)
        are Taylor series in \(\nu_0, ..., \nu_m \). For instance with 3 coefficients, the
        attribut `dcoef[n]` represents the array \((a_n)_{i,j,k}\)
        $$a_n(\nu_0, \nu_1, \nu_2) = \sum_{i,j,k} (a_n)_{i,j,k}  \nu_0^i  \nu_1^j  \nu_2^k$$

        The partial characteristic polynomial reads
        $$ p = \sum \lambda^{n-1} a_{n-1}(\boldsymbol\nu)  + ... + a_0(\boldsymbol\nu)$$

    """

    def __init__(self, dLambda, nu0=None):
        """ Initialize the object with a list of derivatives of the eigenvalue or
            a list of Eig objects.
        """

        if isinstance(dLambda[0], np.ndarray):
            self.dLambda = dLambda
            if nu0 is not None:
                self.nu0 = nu0
            else:
                ValueError('If initialized by a list of array, nu0 is mandatory.')

        elif isinstance(dLambda[0], AbstractEig):
            self.dLambda = [vp.dlda for vp in dLambda]
            self.nu0 = dLambda[0].nu0
        else:
            TypeError('Unsupported type for dLambda elements.')

        # Compute the polynomial coef
        self.dcoefs = self.vieta(self.dLambda)
        for i, c in enumerate(self.dcoefs):
            self.dcoefs[i] = np.asfortranarray(div_factorial(c))

        # Compute usefull coefficients for jacobian evaluation
        self._jacobian_utils()

    def __repr__(self):
        """ Define the representation of the class
        """
        return "Instance of {}  @nu0={} with #{} derivatives.".format(self.__class__.__name__,
                                                                      self.nu0,
                                                                      self.dLambda[0].shape)

    @staticmethod
    def vieta(dLambda):
        """Compute the sucessive derivatives of the polynomial coefficients knowning
        its roots and their successive derivatives.

        The methods is based on the Vieta formulas see for instance
        [Vieta' formulas](https://en.wikipedia.org/wiki/Vieta%27s_formulas)

        Parameters
        ----------
        dLambda : list
            List of the polynom roots successive derivatives. The input format is
            [D**(alpha) lda1, D**(alpha) lda2, ...], where alpha represent the
            multi-index with the derivation order for each variable.
            D**(alpha) lda1 is an ndarray and the first element [0, ...] is lda1.
            All D**(alpha) lda_i should have the same shape.

        Returns
        -------
        dcoef_pol : list
            list of the derivatives of each coef of the polynom. It is assumed that
            a_N=1. The coefficaients are sorted in decending order such,
            1 lda**N + ... + a_0 lda**0

        Examples
        --------
        Consider x**2 - 3*x + 2 = (x-1)*(x-2), where all roots are constant (0 derivatives)
        >>> dlambda=[np.array([[2, 0], [0, 0]]), np.array([[1, 0], [0, 0]])]
        >>> c = CharPol.vieta(dlambda)
        >>> norm(np.array([c[0].flat[0], c[1].flat[0], c[2].flat[0]]) - np.array([1, -3, 2])) < 1e-10
        True

        """
        # FIXME probably a problem if the number of lda is not even

        # Number of derivative is N - 1
        N = np.array(dLambda[0].shape) - 1
        # M number of eigenvalue in the set
        M = len(dLambda)
        # Intialize the coef of the polynomial
        dcoef_pol = []
        dcoef0 = np.zeros_like(dLambda[0], dtype=np.complex)
        dcoef0.flat[0] = 1.
        dcoef_pol.append(dcoef0)

        # Loop to create each coefficients of the polynomial depeding of the order
        for order in range(1, M+1):
            lda_id = np.arange(0, M)
            # local coeff for summation
            dcoef_pol_ = np.zeros_like(dLambda[0], dtype=np.complex)

            # TODO add test if no product ?
            # get all combinaison of with 1 eigenvalue , 2 etc....
            for index in it.combinations(lda_id, order):
                # Compute the values derivative of the product
                dcoef_pol_ += diffprodTree([dLambda[i] for i in index], N)

            dcoef_pol.append(dcoef_pol_*(-1)**order)

        # Return the successive derivative of the polynomial coef, no factorial !!
        return dcoef_pol

    def EP_system(self, vals):
        """ Evaluate the successive derivatives of the partial characteristic
        polynomial with respect to lda.

        This system is built on the sucessive derivatives of the partial
        characteristic polynomial,

        v = [p0, p1, ... pn]^t, where pi = d^ip0/dlda^i.

        Generically, this système yields to a finite set of EP.

        # TODO need to add a test (validated in toy_3doy_2params)
        # may be also obtain with the jacobian matrix....

        Parameters
        ----------
        vals : iterable
            Containts the value of (lda, nu_0, ..., nu_n) where the polynom
            must be evaluated. Althought nu is relative to nu0, absolute value
            have to be used.

        Returns
        -------
        v : np.array
            Value of the vectorial function @ vals
        """
        # Total number of variables (N-1) nu_i + 1 lda
        N = len(vals)
        # polynomial degree in lda
        deg = len(self.dcoefs)
        # Extract input value
        lda, *nu = vals
        nu = np.array(nu, dtype=np.complex) - np.array(self.nu0, dtype=np.complex)
        # compute the an coefficients at nu
        an = np.zeros((len(self.dcoefs),), dtype=np.complex)
        # Compute a_n at nu
        for n, a in enumerate(self.dcoefs):
            # an[n] = pu._valnd(polyval, a, *nu)
            an[n] = polyvalnd(nu, a)

        # Compute the derivtaive with respect to lda

        # Create a coefficient matrix to account for lda derivatives
        # [[1, 1, 1, 1], [3, 2, 1, 1], [2, 1, 1, 1]
        DA = np.ones((N, deg), dtype=np.complex)
        for i in range(1, N):
            DA[i, :-i] = np.arange((deg-i), 0, -1)
        # Evaluate each polynom pi
        v = np.zeros((N,), dtype=np.complex)
        for n in range(0, N):
            # apply derivative with respect to lda
            dan = an[slice(0, deg-n)] * np.prod(DA[0:(n+1), slice(0, deg-n)], 0)
            # np.polyval start by high degree
            # np.polynomial.polynomial.polyval start by low degree!!!
            v[n] = polyvalnd(lda, dan[::-1])

        return v

    def _jacobian_utils(self):
        """ Precompute the shifted indices and combinatoric elements used
        in jacobian matrix (N+1) x (N+1).

        Attributes
        ----------
        _dlda_prod : np.array
            Coefficients used in 1D (lda) polynomial derivative. Assume decending order.
        _der_coef : np.array
            Coefficients used in ND (nu_0, ..., nu_N) polynomial derivative.
        _da_slice : np.array
            List of slices of da array.
        _da_shape : np.array
            List of shape of da array.
        """
        # get number of Parameters
        N = len(self.dcoefs[0].shape) + 1
        # polynomial degree in lda
        deg = len(self.dcoefs)
        # shape of a(nu)
        shape = np.array(self.dcoefs[0].shape)
        # Create a coefficient matrix dlda_prod to account for lda derivatives
        # [[1, 1, 1, 1], [3, 2, 1, 1], [2, 1, 1, 1]
        DA = np.ones((N+1, deg), dtype=np.complex)
        for i in range(1, N+1):
            DA[i, :-i] = np.arange((deg-i), 0, -1)
        self._dlda_prod = np.cumprod(DA, 0)

        # Create Coefficients used nu_i derivatives
        self._der_coef = np.empty((N, N), dtype=np.object)
        self._da_slice = np.empty((N, N), dtype=np.object)
        self._da_shape = np.empty((N, N), dtype=np.object)
        # Loop of the derivative of P, fill J raws
        for row in range(0, N):
            # loop over the nu variables, fill column
            for col in range(0, N):
                # store coef for lda-evaluation
                # recall that an[0] * lda**n
                an = np.zeros((len(self.dcoefs),), dtype=np.complex)
                # create the matrix accounting for derivative of coefs
                der_coef_list = []
                start = np.zeros_like(shape)
                # shape of the derivatives of the coef
                da_shape = shape.copy()
                if col > 0:
                    da_shape[col-1] -= 1
                    start[col-1] += 1
                da_slice_list = []
                for nu_i in range(0, N-1):
                    if nu_i == col - 1:
                        der_coef_list.append(np.arange(1, da_shape[nu_i]+1))
                        da_slice_list.append(slice(1, da_shape[nu_i]+1))
                    else:
                        der_coef_list.append(np.ones((da_shape[nu_i],)))
                        da_slice_list.append(slice(0, da_shape[nu_i]))
                # Maintains fortran ordering
                self._der_coef[row, col] = np.asfortranarray(_outer(*der_coef_list))
                self._da_slice[row, col] = tuple(da_slice_list)
                self._da_shape[row, col] = da_shape.copy()

    def jacobian(self, vals):
        """ Compute the jacobian matrix of the EP system at nu.

        This system is built on the sucessive derivatives of the partial
        characteristic polynomial,

        J = [[dp0/dlda, dp0/dnu_0 ...., dp0/dnu_n],
             [dp1/dlda, dp1/dnu_0 ...., dp1/dnu_n],
              .....]
        where pi = d^ip0/dlda^i.

        Generically, this système yields to a finite set of EP.

        Althought nu is relative to nu0, absolute value have to be used.

        # TODO need to add a test (validated in toy_3doy_2params)

        Parameters
        ----------
        vals : iterable
            Containts the value of (lda, nu_0, ..., nu_n) where the polynom
            must be evaluated. Althought nu is relative to nu0, absolute value
            have to be used.

        Returns
        -------
        J : np.array
            The jacobian matrix.

        Notes
        -----
        Works with _jacobian_utils method. This methods will create some required
        indices and constants array.

        """
        # Total number of variables (N-1) nu_i + 1 lda
        N = len(vals)
        # polynomial degree in lda
        deg = len(self.dcoefs)
        # Extract input value
        lda, *nu = vals
        nu = np.array(nu, dtype=np.complex) - np.array(self.nu0, dtype=np.complex)
        J = np.zeros((N, N), dtype=np.complex)

        # Alias coefficient matrix dlda_prod to account for lda derivatives
        dlda_prod = self._dlda_prod
        # Alias coefficients and indices matrices to account for nu derivatives
        der_coef = self._der_coef
        da_slice = self._da_slice
        da_shape = self._da_shape

        # Loop of the derivative of P, fill J raws
        for row in range(0, N):
            # loop over the nu variables, fill column
            for col in range(0, N):
                # store coef for lda-evaluation
                # recall that an[0] * lda**n
                an = np.zeros((len(self.dcoefs),), dtype=np.complex)
                # Loop over the polynomial coefs
                for n, a in enumerate(self.dcoefs):
                    # Recall that a[0,...,0] * nu0**0 * ... * nu_m**0
                    # Create a zeros matrix
                    # Maintains fortran ordering
                    da = np.zeros(da_shape[row, col], dtype=np.complex, order='F')
                    # and fill it with the shifted the coef matrix
                    da = a[da_slice[row, col]] * der_coef[row, col]
                    # an[n] = pu._valnd(polyval, da, *nu)
                    an[n] = polyvalnd(nu, da)
                # apply derivative with respect to lda
                if col == 0:
                    # Increase the derivation order
                    # dan = an[0:-(row+1)] * np.prod(DA[1:(row+2), :-(row+1)], 0)
                    dan = an[0:-(row+1)] * dlda_prod[row+1, :-(row+1)]
                else:
                    # Apply successived derivative of the parial Char pol
                    # dan = an[slice(0, deg-row)] * np.prod(DA[0:(row+1), slice(0, deg-row)], 0)
                    dan = an[slice(0, deg-row)] * dlda_prod[row, slice(0, deg-row)]
                    # np.polyval start by high degree
                    # np.polynomial.polynomial.polyval start by low degree!!!
                # J[row, col] = polyval(lda, dan[::-1])
                J[row, col] = polyvalnd(lda, dan[::-1])
        return J

    def eval_at(self, vals):
        """ Evaluate the partial caracteristic polynomial at (lda, nu).

        Parameters
        ----------
        vals : iterable
            Containts the value of (lda, nu_0, ..., nu_n) where the polynom
            must be evaluated. Althought nu is relative to nu0, absolute value
            have to be used.

        Returns
        -------
        The partial caracteristic polynomial at (lda, nu).

        # TODO need to add a test (validated in toy_3doy_2params)
        """
        # Extract input value
        lda, *nu = vals
        # Evaluate the polynomial *coefficient* at nu
        an = self._eval_an_at(nu)
        # Evaluate the polynomial
        # np.polyval start by high degree
        # np.polynomial.polynomial.polyval start by low degree!!!
        return polyvalnd(lda, an[::-1])

    def _eval_an_at(self, nu):
        """ Evaluate the partial caracteristic polynomial coefficient an at nu.

        Parameters
        ----------
        nu : iterable
            Containts the value of (nu_0, ..., nu_n) where the polynom
            must be evaluated. Althought nu is relative to nu0, absolute value
            have to be used.

        Returns
        -------
        an : array
            The partial caracteristic polynomial coefficient at nu in descending order
            an[0] * lda **(n-1) + ... + an[n-1]
        """
        # Extract input value
        nu = np.array(nu, dtype=np.complex) - np.array(self.nu0, dtype=np.complex)
        an = np.zeros((len(self.dcoefs),), dtype=np.complex)
        # Compute a_n at nu
        for n, a in enumerate(self.dcoefs):
            # an[n] = pu._valnd(polyval, a, *nu)
            an[n] = polyvalnd(nu, a)

        # np.polyval start by high degree
        # np.polynomial.polynomial.polyval start by low degree!!!
        return an


    @staticmethod
    def _newton(f, J, x0, tol=1e-8, normalized=True, verbose=False):
        """ Basic Newton method for vectorial function.

        Parameters
        ----------
        f : function
            The vectorial function we want to find the roots. Must Returns
            an array of size N.
        J : function
            Function that provides the Jacobian matrix NxN.
        x0 : array
            The initial guess of size N.
        tol : float
            The tolerance to stop iteration.
        normalized : bool
            If `True` the tol is applied to the ratio of the ||f(x)||/||f(x0)||
        verbose : bool
            print iteration log.


        Returns
        -------
        x : array
            The solution or None of NiterMax has been reach or if the
            computation fail.
        """

        # set max iteration number
        NiterMax = 150
        x = np.array(x0)
        k = 0
        cond = True

        while cond:
            fx = f(x)
            if k == 0:
                norm_fx0 = np.linalg.norm(fx)
            # working on polynomial may leads to trouble if x is outside the
            # convergence radius. Use try/except
            try:
                x = x - np.linalg.inv(J(x)) @ fx
                if verbose:
                    print(k, x, abs(fx))
                k += 1
                # condition
                if normalized:
                    cond = (k < NiterMax) and (np.linalg.norm(fx)/norm_fx0) > tol
                else:
                    cond = (k < NiterMax) and np.linalg.norm(fx) > tol
            except:
                print('Cannot compute Newton iteration at x=', x)
                print('  when starting from x0=', x0)
                return None

        # if stops
        if k < NiterMax:
            return x
        else:
            return None

    def newton(self, bounds, Npts=5, decimals=6, max_workers=4, tol=1e-4, verbose=False):
        """ Mesh parametric space and run newton search on each point in
            parallel.

        bounds : iterable
            each item must contains the 2 bounds in the complex plane. For
            instance if bounds = [(-1-1j, 1+1j), (-2-2j, 2+2j)],
            the points will be put in this C**2 domain.
            Althought nu is relative to nu0, absolute value have to be used.
        Npts : int
            The number of point in each direction between the bounds.
        decimals : int
            The number of decimals keep to filter the solution
        max_workers : int
            The number of worker to explore the parametric space.
        normalized : bool
            If `True` the tol is applied to the ratio of the ||f(x)||/||f(x0)||
        verbose : bool
            print iteration log.

        Returns
        -------
        sol : array
            The N 'unique' solutions for the M unknowns in a NxM array.
        """
        # TODO May limit bounds by convergence radius ?

        # Create a coarse mesh of the parametric space for Newton solving
        grid = []
        for bound in bounds:
            grid.append(np.linspace(bound[0], bound[1], Npts))

        # For each, run Newton search
        all_sol = []
        # use partial to fixed all function parameters except lda
        _p_newton = partial(self._newton, self.EP_system, self.jacobian, tol=tol,
                            verbose=verbose)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for s in executor.map(_p_newton,
                                  it.product(*grid),
                                  chunksize=1):
                all_sol.append(s)

        # Filter all solution to keep only unique
        # Remove None and convert to array
        sol_ = np.array([s for s in all_sol if s is not None])
        # Use unique to remove duplicated row
        sol = np.unique(sol_.round(decimals=decimals), axis=0)

        return sol

    @staticmethod
    def taylor2sympyPol(coef_list, tol=1e-12):
        """ Convert a polynom a_n(nu) lda**n + .... + a_0(nu),
        where nu stands for all variables, into a sympy polynomial
        object in (nu, lda)

        Parameters
        ----------
        coef_list : List
            Contains ndarray with the Taylor series of an [an, a_n-a, ..., a0]

        Returns
        -------
        P : Pol
            Multivariate polynomial in sympy format on complex field.
        tol : float
            Tolerance below which coefficients are set to 0 in sympy polynomials.

        Examples
        --------
        >>> a2, a1, a0 = np.array([[1, 2], [3, 6]]), np.array([[10, 0], [0, 0]]), np.array([[0, 5], [7, 0]])
        >>> P, variables = CharPol.taylor2sympyPol([a2, a1, a0])  # lda**2 a2 + lda * a1 + a0
        >>> print(P)
        Poly(6.0*lambda**2*nu0*nu1 + 3.0*lambda**2*nu0 + 2.0*lambda**2*nu1 + 1.0*lambda**2 + 10.0*lambda + 7.0*nu0 + 5.0*nu1, lambda, nu0, nu1, domain='CC')
        """
        # TODO is there a best order ?
        # FIXME add a truncation flag to sparsify
        coef_dict = dict()
        N = len(coef_list)
        # Store all coefs in a dict²
        for n, c in enumerate(reversed(coef_list)):
            for index, dc in np.ndenumerate(c):
                if abs(dc) > tol:
                    key = (n, *index)
                    coef_dict[key] = dc

        # get the number of Parameters nu
        nu_dim = len(coef_list[0].shape)
        var_string = 'lambda, nu0:' + str(nu_dim)
        # Use the dict to create a sympy Poly expression
        variables = sym.symbols(var_string)
        P = sym.Poly(coef_dict, variables, domain='CC')

        return P, variables

    def groebner_solve(self, tol):
        """ Compute Groebner bases to solve find the EP of higher degree.

        Parameters
        ----------
        tol : float
            Tolerance below which coefficients are set to 0 in sympy polynomials.

        # TODO recast into EP object ?

        Returns
        -------
        ep : List

        """
        # TODO add Nmax...
        # Create the Ideal form by partial characteristic and its derivative
        ideals = []
        p0, variables = self.taylor2sympyPol(self.dcoefs, tol)
        print(p0)
        ideals.append(p0)
        # FIXME : Assume lda is the first in lex order
        lda = variables[0]
        # Create an ideals using #variables derivaties of p0
        for i in range(1, len(p0.degree_list())):
            ideals.append(ideals[i-1].diff(lda))
        # Use lex ordering since it has elimination property
        g = sym.groebner(ideals, method='f5b', domain='CC', order='lex')
        # solve numerically
        variables = list(g.free_symbols)
        variables.sort(key=str)
        if g.is_zero_dimensional:
            # see ideals, varieties and Algorithms, p. 230
            ep = CharPol._rec_solve(g, variables)
        else:
            print('Warning: g is not 0 dimensional.')
            ep = None

        return ep, p0, g, ideals

    @staticmethod
    def _rec_solve(g, variables, nu=None, sol=None):
        """ Recursive elimination of the Groebner bases for the variables
        contains in `variables`.

        # FIXME not fully working !!
        - need to check 0 after substitution
        - need to check if wrong solution occured ex: 2=0

        Parameters
        ----------
        g : list/grobener basis
            Groebner basis
        variables : list
            ordered list of groebner basis variables
        nu : list
            Fixed variables used in the elimitation process. Should be None,
            used internally in the recursion.
        sol : deque
            Should be None, used internally in the recursion.

        Returns
        -------
        sol : deque
            All the solutions given in the order provided by `variables`.

        Exemples
        --------
        # Based on sympy `nonlinsolve` examples
        # >>> x, y = sym.symbols('x, y')
        # >>> F = [sym.poly(x**2 - 2*y**2 - 2, x,y, domain='CC'), sym.poly(x*y - 2, x,y, domain='CC')]
        # >>> g = sym.groebner(F, method='f5b', domain='CC')
        # >>> sol = CharPol._rec_solve(g, [x, y])

        # The references solution from sympy doc (symbolic)
        # FiniteSet((-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I))
        # >>> sol_ref = [[-2, -1], [2, 1], [-np.sqrt(2)*1j, np.sqrt(2)*1j], [np.sqrt(2)*1j, -np.sqrt(2)*1j]]

        # Check if the two sets are identical up to round-off error, using complex number
        # for ordering each set.
        # >>> sol_c = np.sort(np.array([a[0]+1j*a[1] for a in sol]))
        # >>> sol_ref_c = np.sort(np.array([a[0]+1j*a[1] for a in sol_ref]))
        # >>> np.linalg.norm(sol_c - sol_ref_c) < 1e-10
        # True


        Based on Ideals, Varieties and algorithms, p. 122
        >>> x, y, z = sym.symbols('x, y, z')
        >>> F = [sym.Poly(x**2 + y + z - 1, x, y, z, domain='CC'), sym.Poly(x + y**2 + z -1, x,y,z, domain='CC'), sym.Poly(x + y + z**2 -1, x,y,z, domain='CC')]
        >>> g = sym.groebner(F, method='f5b', domain='CC')
        >>> sol = CharPol._rec_solve(g, [x, y, z])

        """
        # DEBUG
        if len(g) == 4:
            print('here we are')

        # At FIRST call :
        # Create the solution list if needed
        if sol is None:
            sol = []
        # Convert Groebner Object into list  (easier to process)
        if g is not isinstance(g, sym.polys.polytools.GroebnerBasis):
            g = list(g)

        # Get the number of variables
        N = len(variables)

        # Process the remaining part of g...
        cond = True  # be sur to enter in the while loop
        r_list = []
        # several groebner basis polynomials may be univariate
        # we loop on them
        while cond:
            # Extract the last univariate polynomial
            g_ = g.pop()
            # create a poly since 'subs' destroy poly.
            # use free_symbols to find the involved variables
            last_g = sym.poly(g_, g_.free_symbols, domain='CC')
            # Convert it into numpy univariate polynomial format
            # use all_coeffs else missing 0 coefs for numpy roots
            if len(last_g.free_symbols) > 0:
                c = np.array(last_g.all_coeffs(), dtype=np.complex)
            else:
                # No solution, return without modification
                return sol
            # Compute its the roots **numerically**
            # ri = np.roots(c)
            ri = np.array(list(sym.roots(last_g).keys()))
            r_list.append(ri)
            print(last_g.gens, ri)
            if len(g) != 0:
                cond = last_g.free_symbols == g[-1].free_symbols
            else:
                cond = False
        # contains all the roots of all univariates polynomials with the same variables
        r = np.concatenate(r_list)
        # Loop over all the roots and process recursivelly
        # the next polynomials
        for ri in r:
            # Appends the already knwon variables
            if nu is None:
                nui = [ri]
            else:
                # nui is fill from the left side (since elimination is done
                # in reversed order)
                nui = [ri, *nu]

            if len(nui) == N:
                # End of recurence
                # Appends new solutions
                sol.append(nui)
            else:
                # Recursive call for next polynomial
                # Create a dict with already knwon variables
                variables_ = {variables[-(i+1)]: nu_ for i, nu_ in enumerate(nui)}
                # Remarks : subs destroy poly and create 'add' object
                gi = [gi_.subs(variables_) for gi_ in g]
                sol = CharPol._rec_solve(gi, variables, nui, sol)
        return sol

    def getdH(self):
        r""" Compute the Taylor series of H function with respect to nu.
        
        H is proportional to the discriminant of the partial characteristic polynomial.

        This method provides Taylor series of Discriminant, whereas the `discriminant`
        method, using the Sylvester matrix, just provides an evaluation at fixed
        nu values.
        This Taylor series may be particularly usefull when dealing with univariate
        polynomial to find all the roots.

        This _discriminant_ estimation is obtained from the eigenvalues (roots of the CharPol),
        $$
        H(\nu) = \prod_{i<j} (\lambda_i(\nu)-\lambda_j(\nu))^2.
        $$
        this expression can be obtained, seeing that
        $$
        H(\nu) = \prod_{p \in P} h_{p}(\nu)
        $$
        Here, we called \(P\) the set of all possible pair of eigenvalues from the set of the
        selected eigenvalue \(\Lambda\).


        Returns
        -------
        TH : list
            Contains the Taylor series of H.

        Remarks
        -------
        H is equal to the discriminant iff an=1, else need to multiply by an**(2*n -2).
        """

        dLambda = self.dLambda

        # assume all dLambda have the same shape
        N = np.array(dLambda[0].shape) - 1     # Number of derivative
        M = len(dLambda)                       # M number of eigenvalue in the set
        # Card of permution of 2 in M element
        # CM = int(sp.special.binom(M, 2))
        lda_id = np.arange(0, M)

        Index = []
        # size CM, contains derivative h(i pair)
        dh = []
        # Get all combinaison of eigenvalue and compute dh(i pair)
        for i in it.combinations(lda_id, 2):
            # strore index for later use
            Index.append(i)
            # Compute the values of dh for a pair :
            # f = (lda0 -  lda1)**2 = (lda0 -  lda1) * (lda0 -  lda1)
            f = dLambda[i[0]].squeeze() - dLambda[i[1]].squeeze()
            dh_ = diffprodMV([f, f], N)
            # use dedicated formulas for 1 variable, (less sumation)
            # dh_ = EP.dlda2dh(dLambda[i[0]].squeeze(),
            #                  dLambda[i[1]].squeeze())
            dh.append(dh_)

        # Derivate the product of h_i
        dH = diffprodTree(dh, N)
        # dH is the successive derivative, div by factorial to get TH
        TH = Taylor(div_factorial(dH), self.nu0)
        return TH

    def discriminant(self, nu):
        r""" Compute the discriminant from the coefficient and the Sylvester matrix.


        If we consider a polynom p, with coefficients an, the discriminant is given
        $$
        Disc_{x}(p)={\frac {(-1)^{n(n-1)/2}}{a_{n}}}\operatorname {Res}_{x}(p,p')
                   = \frac {(-1)^{n(n-1)/2}}{a_{n}} \det S
        $$ where S is the sylvester matrice.

        Parameters
        ----------
        nu : iterable
            Contains the value (nu_0, nu_1, ...) where the discriminant is computed

        Returns
        -------
        d : complex
        """
        # polynomial coef
        an = self._eval_an_at(nu)
        n = an.size - 1
        dan = an[:-1] * self._dlda_prod[1, :-1]
        # Sylvester need ascending order
        S = self.sylvester(an[::-1], dan[::-1])
        # S = self.sylvester(an, dan)
        d = np.linalg.det(S) * (-1)**(n * (n-1)/2) * an[0]
        return d

    @staticmethod
    def sylvester(p, q):
        r""" Compute the Sylvester matrix associated to p and q.

        https://en.wikipedia.org/wiki/Sylvester_matrix
        Formally, let p and q be two nonzero polynomials, respectively of degree m and n, such
        $$p(z) = p_0 + p_1 z + p_2 z^2 + \cdots + p_m z^m,$$
        $$q(z) = q_0 + q_1 z + q_2 z^2 + \cdots + q_n z^n.$$

        If the two polynomials have a (non constant) common factor. In such a case, the
        determinant of the associated Sylvester matrix, also called the resultant of
        the two polynomials vanishes. If q = dp / dz, the determinant of the Sylvester
        matrix is the discriminant.

        The Sylvester matrix associated to p and q is (n + m) × (n + m) is build
        on coefficients shift as shown in the example below.

        Parameters
        ----------
        p : array
            The coefficient of the polynomial p, sorted in ascending order.
        q : array
            The coefficient of the polynomial q, sorted in ascending order.

        Returns
        -------
        S : array
            Containing the Sylvester matrix (complex)

        Examples
        --------
        p = p = (z-2)**2*(z-3), with a double roots 2
        >>> p = np.array([-12, 16, -7, 1])
        >>> q = np.array([16, -14, 3])
        >>> CharPol.sylvester(p, q).real
        array([[  1.,  -7.,  16., -12.,   0.],
               [  0.,   1.,  -7.,  16., -12.],
               [  3., -14.,  16.,   0.,   0.],
               [  0.,   3., -14.,  16.,   0.],
               [  0.,   0.,   3., -14.,  16.]])
        >>> np.abs(np.linalg.det(CharPol.sylvester(q, p))) < 1e-12
        True

        """
        m = p.size
        n = q.size
        # size of the Symvester matrix
        N = m + n - 2
        S = np.zeros((N, N), dtype=np.complex)
        # if q has higher degree, need to swap them
        if n > m:
            p, q = q, p
            m, n = n, m
        # fill upper part with p coefficients
        for i in range(0, n - 1):
            S[i, i:(i+m)] = p[::-1]
        # fill lower part with q coefficients
        for i in range(0, m - 1):
            S[i+n-1, i:(i+n)] = q[::-1]

        return S



class Taylor:
    """ Define a multivariate Taylor series.
    """

    def __init__(self, an, nu0):
        """ Initialize the object with df/d nu divided by factorial.
        """
        self.an = an
        self.nu0 = nu0

    def __repr__(self):
        """ Define the representation of the class
        """
        return "Instance of {} @nu0={} with #{} derivatives.".format(self.__class__.__name__,
                                                                     self.nu0,
                                                                     self.an.shape)

    def eval_at(self, nu):
        """ Evaluate the Taylor series at nu with derivatives computed at nu0.

        Parameters
        ----------
        nu : iterable
            Containts the value of (nu_0, ..., nu_n) where the polynom
            must be evaluated. Althought nu is relative to nu0, absolute value
            have to be used.

        Returns
        -------
        The evaluation of the Taylor series at nu.
        """
        # Extract input value
        nu = np.array(nu, dtype=np.complex) - np.array(self.nu0, dtype=np.complex)
        # Evaluate the polynomial
        return polyvalnd(nu, self.an)


    # def myroots(coef):
    #     """Provide a global interface for roots of polynom with variable coefficients.

    #     The polynomial is defined such, coef[0] x**n + ... + coef[-1] and


    #     Analytic formula are used for qudratric and cubic polynomial and numerical
    #     methods are used for higher order.


    #     Parameters
    #     ----------
    #     coef : list of complex array like
    #         Contains the value of each coefficients.

    #     Returns
    #     -------
    #     r : list
    #         list of roots for each vector coef

    #     """

    #     # number of coef of the polynome
    #     n = len(coef)
    #     if n == 3:
    #         # quadratique
    #         r = quadratic_roots(*coef)
    #     elif n==4:
    #         # cubic
    #         r = cubic_roots(*coef)
    #     elif n==5:
    #         # quatic, only exceptionhandle is the Biquadratic equation
    #         r = quartic_roots(*coef)

    #     else:
    #         # quelquonque
    #         N = len(coef[0])
    #         r=np.zeros((n-1, N), dtype=np.complex)

    #         for i, c in enumerate(zip(*coef)):
    #             print(c)
    #             ri = np.roots(c)
    #             r[:,i] = ri

    #     return r


    # def quadratic_roots(a, b, c):
    #     """Analytic solution of a quadratic polynom P(x)=ax^2+bx+c.

    #     Parameters
    #     ----------
    #     a, b, c : complex array like
    #         coeff of the polynom such

    #     Returns
    #     --------
    #     r : list
    #         list of roots
    #     """
    #     tol = 1e-8
    #     # check on 1st coef
    #     if np.linalg.norm(a)< tol:
    #         raise ValueError('The first coefficient cannot be 0.')

    #     Delta = np.sqrt(b**2 - 4*a*c, dtype=np.complex)
    #     r1 = (- b - Delta)/(2*a)
    #     r2 = (- b + Delta)/(2*a)
    #     r = [r1, r2]

    #     return r

    # def cubic_roots(a,b,c,d):
    #     """Analytic solution of a cubic polynom  P(x)=ax^3+bx^2+cx+d.

    #     Parameters
    #     ----------
    #     a, b, c, d : complex array like
    #         coeff of the polynom such

    #     Returns
    #     --------
    #     r : list
    #         list of roots

    #     Examples
    #     --------
    #     Solve `x**3 - 4.0*x**2 + 6.0*x - 4.0` with complex roots,
    #     >>> coef = [1, -4, 6, -4]
    #     >>> r = np.array(cubic_roots(*coef)); r.sort()
    #     >>> r_ = np.array([2, 1+1j, 1-1j]); r_.sort()
    #     >>> np.linalg.norm(r - r_) < 1e-12
    #     True
    #     """
    #     tol = 1e-8
    #     # check on 1st coef
    #     if np.linalg.norm(a)< tol:
    #         raise ValueError('The first coefficient cannot be 0.')

    #     D0 = b**2 - 3*a*c
    #     D1 = 2*b**3 - 9*a*b*c + 27*(a**2)*d
    #     C = ((D1 + np.sqrt(D1**2 - 4*D0**3, dtype=np.complex))/2.)**(1./3.)
    #     xi = -0.5 + 0.5*np.sqrt(3)*1j
    #     x = []
    #     for k in range(1, 4):
    #         xk = -1/(3*a)*(b + xi**k*C + D0/(xi**k*C))
    #         x.append(xk)
    #     return x

    # def quartic_roots(a,b,c,d,e):
    #     """Analytic solution of a quartic polynom P(x)=ax^4+bx^3+cx^2+dx+e.

    #     Parameters
    #     ----------
    #     a, b, c, d, e : complex array like
    #         coeff of the polynom. Better to unpack list to set them.

    #     Returns
    #     --------
    #     r : array
    #         list of roots

    #     Remarks
    #     --------
    #     Only the biquadratic exception is implemented. Through extensive tests, no
    #     other problems have been found, but...
    #     https://en.wikipedia.org/wiki/Quartic_function

    #     Examples
    #     --------
    #     Solve `x**4 - 7.0*x**3 + 2.0j*x**3 + 18.0*x**2 - 8.0j*x**2 - 22.0*x + 12.0j*x + 12.0 - 8.0j` with complex roots,
    #     >>> coef = [1, -7.0 + 2.0j, 18.0 - 8.0j, -22.0 + 12.0j, 12.0 - 8.0j]
    #     >>> r = np.array(quartic_roots(*coef)); r.sort()
    #     >>> r_ = np.array([1+1j , 2, 1-1j, 3-2j]); r_.sort()
    #     >>> np.linalg.norm(r - r_) < 1e-12
    #     True

    #     Biquadratic equation (degenerate case)
    #     >>> coef = [1, 0, -9 + 2j, 0, -18j]
    #     >>> r = np.array(quartic_roots(*coef)); r.sort()
    #     >>> r_ = np.array([3, -3, 1-1j, -1+1j]); r_.sort()
    #     >>> np.linalg.norm(r - r_) < 1e-12
    #     True
    #     """
    #     tol = 1e-8
    #     # check on 1st coef
    #     if np.linalg.norm(a)< tol:
    #         raise ValueError('The first coefficient cannot be 0.')

    #     Delta0 = c**2 - 3*b*d + 12*a*e
    #     Delta1 = 2*c**3 - 9*b*c*d + 27*b**2*e + 27*a*d**2 - 72*a*c*e
    #     p = (8*a*c - 3*b**2)/(8*a**2)
    #     q = (b**3 - 4*a*b*c + 8*a**2*d) / (8*a**3)
    #     if np.linalg.norm(q)< tol:
    #         # degenerate case of Biquadratic equation
    #         r = quadratic_roots(*[a, c, e])
    #         r[0] = np.sqrt(r[0])
    #         r[1] = np.sqrt(r[1])
    #         r.extend([-r[0], -r[1]])
    #     else:
    #         Q = np.power( 0.5*(Delta1 + np.sqrt(Delta1**2 - 4*Delta0**3, dtype=np.complex)), 1/3., dtype=np.complex)
    #         S = 0.5*np.sqrt(-2*p/3 + 1/(3*a)*(Q + Delta0/Q), dtype=np.complex )


    #         B = -b/(4*a)
    #         Dp = 0.5*np.sqrt(-4*S**2 - 2*p + q/S, dtype=np.complex)
    #         Dm = 0.5*np.sqrt(-4*S**2 - 2*p - q/S, dtype=np.complex)
    #         r = [B - S + Dp,
    #              B - S - Dp,
    #              B + S + Dm,
    #              B + S - Dm]
    #     return r




# %% Main for basic tests
if __name__ == '__main__':
    # run doctest Examples
    import doctest
    doctest.testmod()
    p = np.array([-12, 16, -7, 1])
    q = np.array([16, -14, 3])
    S = CharPol.sylvester(p, q)
    # """[[1, -7, 16, -12, 0],
    #     [0, 1, -7, 16, -12],
    #     [3, -14, 16, 0, 0],
    #     [0, 3, -14, 16, 0],
    #     [0, 0, 3, -14, 16]]"""