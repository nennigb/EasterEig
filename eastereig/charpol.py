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
## Define CharPol class

This class with handle the *partial * characteristic polynomial.
This polynomial is built from Vieta formulas and from the successive
derivatives of the selected set of eigenvalues.
[Vieta' formulas](https://en.wikipedia.org/wiki/Vieta%27s_formulas)



"""
import numpy as np
from numpy import zeros, asarray, eye, poly1d, hstack, r_
from numpy.linalg import norm
from scipy import linalg
import itertools  as it
from  eastereig.eig import AbstractEig
from eastereig.utils import diffprodTree, div_factorial, diffprodMV
import sympy as sym


class CharPol():
    """ Handle the *partial* characteristic polynomial.
                
        This polynomial is built from Vieta formulas and from the successive
        derivatives of the selected set of eigenvalues.
        
    """

    def __init__(self, dLambda, nu0=None):
        """ Initialize the object with a list of derivatives of the eigenvalue or
            a list of Eig objects.
        """

        if isinstance(dLambda[0], np.ndarray):
            self.dLambda = dLambda
            self.nu0 = nu0
        elif isinstance(dLambda[0], AbstractEig):
            self.dLambda = [vp.dlda for vp in dLambda]
            self.nu0 = dLambda[0].nu0
        else:
            TypeError('Unsupported type for dLambda elements.')

        # Compute the polynomial coef
        self.dcoefs = self.vieta(self.dLambda)
        for i, c in enumerate(self.dcoefs):
            self.dcoefs[i] = div_factorial(c)


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
        I = []
        p0, variables = self.taylor2sympyPol(self.dcoefs, tol)
        print(p0)
        I.append(p0)
        # FIXME : Assume lda is the first in lex order
        lda = variables[0]
        # Create an ideals using #variables derivaties of p0
        for i in range(1, len(p0.degree_list())):
            I.append(sym.diff(I[i-1], lda))
        # Use lex ordering since it has elimination property
        g = sym.groebner(I, method='f5b', domain='C', order='lex')
        # solve numerically
        variables = list(g.free_symbols)
        variables.sort(key=str)
        if g.is_zero_dimensional:
            # see Ideals, varieties and Algorithms, p. 230
            ep = CharPol._rec_solve(g, variables)
        else:
            print('Warning: g is not 0 dimensional.')
            ep = None

        return ep, p0

    @staticmethod
    def _rec_solve(g, variables, nu=None, sol=None):
        """ Recursive elimination of the Groebner bases for the variables
        contains in `variables`.

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
        Based on sympy `nonlinsolve` examples
        >>> x, y = sym.symbols('x,y')
        >>> F = [sym.poly(x**2 - 2*y**2 - 2, x,y, domain='CC'), sym.poly(x*y - 2, x,y, domain='CC')]
        >>> g = sym.groebner(F, method='f5b', domain='CC')
        >>> sol = CharPol._rec_solve(g, [x, y])

        The references solution from sympy doc (symbolic)
        FiniteSet((-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I))
        >>> sol_ref = [[-2, -1], [2, 1], [-np.sqrt(2)*1j, np.sqrt(2)*1j], [np.sqrt(2)*1j, -np.sqrt(2)*1j]]

        Check if the two sets are identical up to round-off error, using complex number
        for ordering each set.
        >>> sol_c = np.sort(np.array([a[0]+1j*a[1] for a in sol]))
        >>> sol_ref_c = np.sort(np.array([a[0]+1j*a[1] for a in sol_ref]))
        >>> np.linalg.norm(sol_c - sol_ref_c) < 1e-10
        True
        """
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
        if len(g) != 0:
            # Extract the last univariate polynomial
            # create a poly since 'subs' destroy poly.
            # use free_symbols to find the involved variables
            # use all_coeffs else missing 0 coefs for numpy roots
            g_ = g.pop()
            last_g = sym.poly(g_, g_.free_symbols, domain='CC')
            # Convert it into numpy univariate polynomial format
            c = np.array(last_g.all_coeffs(), dtype=np.complex)
            # Compute its the roots **numerically**
            r = np.roots(c)

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

    def discriminant(self):
        pass

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

