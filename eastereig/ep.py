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
##Define the EP class

Assuming that the associated eigenvalue \(\lambda^*\) is of algebraical multiplicity 2,
the behavior of the two branches of solutions in the vicinity of \(\nu^*\) is
given by a convergent **Puiseux series** [Kato:1980]
$$
\begin{align}
\lambda_+ &= a_0 + a_1\left(\nu - \nu^*\right)^\frac{1}{2} + \sum_{k=2}^\infty a_k  \left(\nu - \nu^*\right)^{\frac{k}{2}},\\
\lambda_- &= a_0 - a_1\left(\nu - \nu^*\right)^\frac{1}{2} + \sum_{k=2}^\infty a_k \left( -1 \right)^k \left(\nu - \nu^*\right)^{\frac{k}{2}},
\end{align}
$$
where \(a_0=\lambda^*\).

The proposed algorithm exploits the knowledge of high order derivatives of two
selected eigenvalues denoted \(\lambda_+\) and  \(\lambda_-\) and calculated at
an arbitrary value \(\nu_0\).
In order to circumvent the branch point singularity two **analytic auxiliary functions** are defined:
$$
\begin{align}
g(\nu) &= \lambda_+ + \lambda_-,\\
h(\nu) &= \left(\lambda_+ - \lambda_- \right)^2.
\end{align}
$$
By construction, these functions are holomorphic in the vicinity of \(\nu^*\)
where eigenvalues coalesce.

The derivative of \(g\) and \(h\) at \(\nu_0\) can be directly obtained from
the derivative of the selected eigenvalues. Applying Liebnitz' rule for product
derivation yields
$$
\begin{align}
%\frac{\ud^{n}g(\nu)}{\ud \nu^n} &= \frac{\ud^{n}\lambda_1(\nu)}{\ud \nu^n} + \frac{\ud^{n}\lambda_{2}(\nu)}{\ud \nu^n} ,\\
%\frac{\ud^{n}h(\nu)}{\ud \nu^n} &= \sum_{k=0}^n \left[ \binom{n}{k} \lambda_1^{n-k}\lambda_{1}^k + \binom{n}{k} \lambda_{2}^{n-k}\lambda_{2}^k -2 \binom{n}{k} \lambda_1^{n-k}\lambda_{2}^k \right]
g^{(n)}(\nu_0) &= \lambda_+^{(n)}(\nu_0) + \lambda_-^{(n)}(\nu_0) ,\label{eq:dg}\\
%\frac{\ud^{n}h(\nu)}{\ud \nu^n} &= \sum_{k=0}^n \left[ \binom{n}{k}\, \lambda_1^{(n-k)}\lambda_{1}^{(k)} + \binom{n}{k}\, \lambda_{2}^{(n-k)}\lambda_{2}^{(k)} -2 \binom{n}{k}\, \lambda_1^{(n-k)}\lambda_{2}^{(k)} \right],
h^{(n)}(\nu_0) &= \sum_{k=0}^{\lfloor \tfrac{n}{2}\rfloor} \binom{n}{k} \, \left(2-\delta_{\tfrac{n}{2}k}\right)  \left(  \lambda_+^{(n-k)} (\nu_0)\lambda_+^{(k)} (\nu_0)+  \lambda_-^{(n-k)}(\nu_0)\lambda_-^{(k)}(\nu_0) \right) -2 \sum_{k=0}^n \binom{n}{k}\, \lambda_+^{(n-k)}(\nu_0)\lambda_-^{(k)}(\nu_0) , \label{eq:dh}
\end{align}
$$

These functions can be approximated, at least locally, via their truncated Taylor series
$$
\begin{align}
\mathcal{T}_g(\nu) &= \sum_{n=0}^N b_n \left(\nu- \nu_0\right)^n,\quad \text{where } b_n=\frac{g^{(n)}(\nu_0)}{n!} ,\label{eq:gTaylor}\\
\mathcal{T}_h(\nu)&= \sum_{n=0}^N c_n \left(\nu - \nu_0\right)^n,\quad \text{where } c_n=\frac{h^{(n)}(\nu_0)}{n!}. \label{eq:hTaylor}
\end{align}
$$

  1. Since h vanished at the EP, ie when \(\lambda_+=\lambda_-\), its roots yields
  the EP in `EP_loc` attirbut
  2. Get the coefficient \(a_0,\,\dots,\, a_{2N}\) of the truncated Puiseux series
  by matching the truncated Taylor expansion of \(T_h\) and \(T_g\) with their expansion
  using the Puiseux series \(P_{\lambda_+}\) and \(P_{\lambda_-}\).
  Store in the `a` attirbut, compute with `getPuiseux`
"""

# from . import lda_func
# from .utils import multinomial_index_coefficients
from scipy.special import binom, factorial
from scipy.optimize import linear_sum_assignment
import numpy as np

# matplotib not mandarotry for computing
try:
    import matplotlib.pyplot as plt
except:
    print('Warning : matplotlib not imported...')



class EP:
    """Class to locate EP and get Puiseux expansion.

    TODO Need a little clean up !

    Depending on the realized computation, here are given the attribbutes...

    Parameters
    ----------
    vp1dla, vp2dlda: iterable
        provides the pair of Eig object you want to merge ex: Eig.dlda

    Attirbutes
    -----------
    EP_loc: list
        The list of the found EP, obtained after `locate`
    a: list
        the list that contains the Puiseux series Coefficients. `a[index]`
        corresponds to `EP_loc[index]`. Obtained after `getPuiseux`
    Th_roots: np.array
        the roots of T_h of order N, obtained after `locate`
    dg: np.array
        the derivatives of g function
    dh: np.array
        the derivatives of the h function
    aposterioriErr: np.array
        the error between N and N-1 roots. Significant only fort the first terms,
        higher order terms may be in wrong order. Obtained after `locate`

    """

    def __init__(self, vp1, vp2):
        """Init method."""
        # check that vp1 and vp2 are compute areound the same point nu0
        if vp1.nu0 != vp2.nu0:
            raise ValueError('Eigenvalues are not computed at the same point')

        self.nu0 = vp1.nu0
        self._vp1dlda = vp1.dlda
        self._vp2dlda = vp2.dlda

    def __repr__(self):
        """Define the object representation."""
        try:
            EP_loc = self.EP_loc
            Err = self.aposterioriErr
        except:
            EP_loc = []
            Err = []
        text = ("Instance of EP class\n#{} EP found merging between {} and"
                " {}\n  > nu0 : {} \n  > (N-1)-Err : {}\n  > EP list :{}")
        return text.format(len(EP_loc), self._vp1dlda[0], self._vp2dlda[0],
                           self.nu0, Err, EP_loc)

    @staticmethod
    def dlda2dh(dlda1, dlda2):
        """Compute the n-th firts derivative of h = (lda_1-lda2)**2 from dlda1 and dlda2.

        arxiv.org/abs/1909.11579 Eq. 7b
        use standard liebnitz rule and a compact version for symmetric terms

        Examples
        --------
        >>> EP.dlda2dh([1,0.5,0.13,0.01],[3,0.1,0.05,0.015])
        array([ 4.   +0.j, -1.6  +0.j,  0.   +0.j,  0.212+0.j])
        """
        if len(dlda1) != len(dlda2):
            raise IndexError("Derivative sequences must have the same size")

        N = len(dlda1)
        # init derivatives of each terms
        dh1 = np.zeros(N, dtype=complex)
        dh2 = np.zeros(N, dtype=complex)
        dh12 = np.zeros(N, dtype=complex)
        # init 1st terms (no derivation)
        dh1[0] = dlda1[0]**2
        dh2[0] = dlda2[0]**2
        dh12[0] = dlda1[0]*dlda2[0]
        # loop for the N-st derivatives
        for n in range(1, N):
            # symetric terms
            stop = int(np.floor(n/2.))
            for j in range(0, stop+1):
                # delta is equal to one only when $$n$$ is even
                delta = (n/2.) == j
                b = binom(n, j)
                dh1[n] = dh1[n] + b*(2 - delta)*dlda1[n-j]*dlda1[j]
                dh2[n] = dh2[n] + b*(2 - delta)*dlda2[n-j]*dlda2[j]

            # cross terms
            for j in range(0, n+1):
                b = binom(n, j)
                dh12[n] = dh12[n] + b*dlda1[j]*dlda2[n-j]
        # assemble all local out
        dh = dh1 + dh2 - 2*dh12
        return dh

    @staticmethod
    def Pmatrix(z0, N):
        r"""Compute the matrix P defined by equating each power of \(\nu\) of the
        Puiseux and Taylor expansions of the function \(g(\nu)\).

        arxiv.org/abs/1909.11579 Eq. 13.

        Parameters
        ----------
        z0: float
            Computation point of the Taylor expansion
        N: int
            Polynomial degree

        Results
        ----------
        P : matrix

        Examples
        --------
        >>> EP.Pmatrix(0.5+0.1j,3)
        array([[ 1.  +0.j , -0.5 -0.1j,  0.24+0.1j],
               [ 0.  +0.j ,  1.  +0.j , -1.  -0.2j],
               [ 0.  +0.j ,  0.  +0.j ,  1.  +0.j ]])

        """
        P = np.zeros((N, N), dtype=complex)
        for i in range(N):
            for j in range(i, N):
                P[i, j] = binom(j, i)*(-z0)**(j-i)
        return P

    @staticmethod
    def solveOddPower(xi, c, N):
        r""" Compute odd terms of the Puiseux expansion using iterative solver
        of the Non-Linear system

        arxiv.org/abs/1909.11579 Eq. 15-16.

        Description
        -----------
        Equating each power of \(\nu'\) gives explicitly the first coefficient for \(k=1\),
        $$
        \begin{equation}
        a_1 = \pm \frac{1}{2} \sqrt{(\mathbf{P}(\nu_0-\nu^*) \mathbf{c})_1},
        \end{equation}
        $$
        where the sign refers to one of two branches of the Puiseux series given in Eq.~\eqref{eq:Puiseux}.
        By convention, the positive root is chosen.

        The others coefficients are obtained iteratively as
        $$
        \begin{equation}
        a_{2k-1} =  \frac{1}{8 a_1} \left\{ (\mathbf{P}(\nu_0-\nu^*) \mathbf{c})_k\, - 4\sum_{\substack{n=3,5,\ldots,2k-3\\ n \ge 2k-3}}
        	a_n \, a_{2k-n} \right\}, \quad \text{for } k=2,3,\ldots.
        \end{equation}
        $$


        Parameters
        ----------
        xi: float
            Distance between the computational point and the EP
        c: (n,) array_like
            Coefficients of the Taylor series of h
        N: int
            Number of term in a0, N<len(c)

        Returns
        -------
        ao: (N,) array_like
            Odd terms of the Puiseux expansion
            N.B. : The correspondance between odd indexes and ao indexes is
                a1    a3    a5     a7     a9  paper
                ao[0]  ao[1]  ao[2]  ao[3]  ao[4] code
        Remarks
        -------
        The last terms ao[N-1] is always zeros, if N = len(c)

        Examples
        ---------
        >>> xi = (0.06272493020365777-0.3406129756156909j)
        >>> dhTay =np.array([ 23.487594308283406  +4.152311636429914j,  23.00877770477632  +39.83161452854898j , 114.78159411784713  +30.800282110858674j,         50.21819439126324  +26.042138882295525j,  50.949270824732665 -31.375132157602042j, -50.1166871854503    -8.07630647383412j ,        -10.072790625494802  -9.01862569778718j , -13.789863546772875 +38.64297753759615j ,  42.84022867603538   -6.420680271278298j,          1.7297803098142044 -8.38361177245119j , -12.80495718663174  -32.96030830977034j , -33.0017158344141   +25.890789141576224j,         14.106202411204107 +19.10887827873383j ,  31.435800399928002 +11.403948384916266j,  11.524175094085809 -45.8376395712606j  ])
        >>> EP.solveOddPower(xi, dhTay, 5)
        array([ 2.80880799+3.83173729j,  2.27245111+1.00335605j,
                4.01803468+0.48898334j,  0.37676432-2.3769505j ,
               -1.45112934-3.49571785j])
        """

        r = EP.Pmatrix(xi, len(c)) @ c

        # check requested number of coefficients
        if N > len(r):
            raise IndexError('Ask for more coef than coef in Taylor series\n')

        if N == len(r):
            # the last terms cannot be computed
            Nmax = N - 1
        else:
            Nmax = N

        # init to 0
        ao = np.zeros((N,), dtype=complex)

        # n = 1, chose positive solution first
        ao[0] = np.sqrt(complex(r[1])/4.)
        alpha = 1./(8*ao[0])

        # n >= 2
        for n in range(1, Nmax):
            S = 0
            if n > 1:
                idx = range(1, n)
                S = 4.*sum([ao[idx[i]]*ao[idx[-1-i]] for i in range(n-1)])
                # for i in range(n-1):
                #     print('n=',n,idx[i],idx[-1-i] )
            ao[n] = (r[n+1] - S)*alpha

        return ao

    def _dh(self):
        """Compute T_h using truncated Taylor of lda1 and lda2."""
        dh = EP.dlda2dh(self._vp1dlda, self._vp2dlda)
        # store
        self.dh = dh
        # store also Taylor coeff
        self._dhTay = dh/factorial(np.arange(len(self.dh)))

    def _dg(self):
        """Compute T_g using truncated Taylor of lda1 and lda2."""
        # compute the derivatives g**(n)
        self.dg = np.array(self._vp1dlda) + np.array(self._vp2dlda)
        # store also Taylor coeff
        self._dgTay = self.dg/factorial(np.arange(len(self.dg)))

    def _roots(self, tronc):
        """Compute the roots of Th and sort it in ascending order of modulus.

        Parameters
        ----------
        tronc : int
            tuncature value. tronc=0 means all N value, tronc=1 means N-1
        Returns
        -------
        roots : array_like
            the roots of Th shifted by nu0
        """
        # compute h derivative and Taylor
        self._dh()
        # get roots  Th_N
        roots = np.roots(self._dhTay[-(1+tronc)::-1])
        ind = np.argsort(np.abs(roots))
        roots = roots[ind] + self.nu0
        return roots

    def locate(self, tol=1e-2, xi=0.95, tronc=0):
        r"""Compute the roots \(\zeta_n\) of \(T_h^{N}\) the taylor expansion of h of order N.

        The Exceptional Point (EP) is/are one of these roots.

        Others roots which do not correspond to the EP are regularly arranged. 
        They mostly seems to be localized on a circle of radius R.

        The algorithm to identified the EP is:
          1. Compare the roots with \(T_h^{N-1}\)
          2. Check if they belong to the mean radius circle \(\vert \zeta_n \vert < \xi R\).

        Parameters
        ----------
        tol: float
            the tolerance value between N and N-1 roots. Default value 1e-2
        a: float
            coef to make more stringent the condition on the mean radius R. Default value 0.95

        Returns
        -------
        EP_loc: list
            the list of exceptional points
        """
        # roots of ThN
        roots = self._roots(tronc=0)
        # roots of Th_{N-1}
        roots1 = self._roots(tronc=1)
        # routh estimate of Th radius of convergence upper bound
        ThRadius = np.abs(roots-self.nu0).mean()

        # locate EP
        # Err = np.ones(roots1.shape)

        # use munkres algorithm to find the correspondance between root and roots1
        # simple sort is not sufficent if roots are complex conjugate (ie same modulus)
        closest = np.abs(roots-self.nu0) < xi*ThRadius
        # compute the distance matrix D between all combination
        R1, R2 = np.meshgrid(roots[closest], roots1[closest[:-1]])
        D = np.abs(R1-R2)
        row_ind, col_ind = linear_sum_assignment(D)
        Err = D[row_ind, col_ind]

        EPlist = np.where(Err < tol)[0]
        self.aposterioriErr = Err[EPlist].tolist()
#                print('> EP Error vs N and N-1 = %f\n' % (abs(EP_locN1-EP_loc)/abs(EP_loc)))
        self.EP_loc = roots[EPlist].tolist()
        self.Th_roots = roots

        return self.EP_loc

    def getPuiseux(self, index=0):
        r"""Get Puiseux series coefficient at selected EP by `index`.

        Compute first even and then odd terms.

        Parameters
        ----------
        index : int
            the index of the selected EP
        Returns
        -------
        a : a list
            a[index] contains the puiseux series coef

        Remarks
        -------
        \(T_g\) is computed at \(\nu_0\) and the Puiseux series are given at \(\nu^*\).
        To limit the condition number of \(\mathbf{P}(\nu^*)\) when \(\vert \nu^*\vert \gg 1\),
        in Eq. (11) of [arxiv:1909.11579],
        it is preferable to make the change of variable \( \nu'=\nu-\nu^* \).
        This yields to $$ 2 \mathbf{a}_e = \mathbf{P}(\nu_0-\nu^*) \mathbf{b},$$ 
        this expression avoids to inverse \(\mathbf{P}(\nu^*)\).
        """
        # init and check
        try:
            EP_loc = self.EP_loc[index]
        except:
            EP_loc = self.locate()[index]

        # compute derivatives of g (sum)
        self._dg()

        # sure dh exist
        try:
            self.dh
        except:
            self._dh()

        # number of derivatives
        N = len(self.dg)

        # Even coefficients of the puiseux series
        # Pmat0 = np.eye( N, dtype=complex)
        PmatDeltaParam0_EP = EP.Pmatrix((self.nu0-EP_loc), N)
        dgTayCoef = self.dg / factorial(np.arange(N))
        # if Pmat0 = eye, no need inversion
        # ae0 =  np.dot( Pmat0 ,(np.dot(PmatDeltaParam0_EP,dgTayCoef)))/2.
        ae0 = (PmatDeltaParam0_EP @ dgTayCoef)/2.

        # odd coefficients of the puiseux series
        ao = EP.solveOddPower((self.nu0-EP_loc), self._dhTay, N)

        # concatenation of the 2 coefs. famillies
        self.a = [None]*len(self.EP_loc)
        # Coeff of the Puiseux serie of the first eigenvalue (+ sign)
        self.a[index] = np.concatenate([(ae0[i], ao[i]) for i in range(len(ae0))])

        return self.a

    def plotZeros(self, index=0, Title='empty', Couleur='k',
                  variable='\\nu', fig=-1):
        """
        Plot the roots of T_h and compute it if it has not been down.

        The Exceptional Point (EP) is one of these roots. Others roots which do not
        correspond to the EP are regularly arranged. They mostly seems to be
        localized on a circle.
        """
        # FIXME change the name !
        # check if EP have been found
        try:
            EP_loc = self.EP_loc
        except:
            EP_loc = self.locate()

        def labelShift(G, X, shift):
            """Compute the label shift to avoid overlay. if there is only on value, the angle is 0.

            First compute center of gravity G of the cluster of points, then
            place label along GX[i] using `labelShift`.

            Parameters
            ----------
            G : complex
                center of gravity of the cluster of roots
            X : complex
                the roots to labeled
            shift : float
                modulus of the complex shift
            """
            angle = np.angle(G-X)
            cshift = -shift*np.exp(1j*angle)
            return cshift

        # number of EP found
        NEP = max(len(self.EP_loc), 1)
        zeros = self.Th_roots
        nu0 = self.nu0
        # create circle guide for eyes
        rcircle = np.mean(abs(self.Th_roots[NEP-1::]-nu0))
        theta = np.linspace(0, 2*np.pi, 101)
        circle = np.exp(1j*theta)*rcircle + self.nu0

        figTaylor = plt.figure(fig)

        if Title != 'empty':
            figTaylor.canvas.set_window_title(Title)
        plt.plot(circle.real, circle.imag, '-.')
        # Th_N roots
        plt.scatter(zeros.real[0::], zeros.imag[0::], marker='.', s=80, color=Couleur)
        plt.scatter(nu0.real, nu0.imag, marker='x', s=80, color=Couleur)
        # Th_N-1 roots
        zeros1 = self._roots(tronc=1)
        plt.scatter(zeros1.real[0::], zeros1.imag[0::], marker='o',
                    facecolors='none', edgecolors=Couleur, s=80, color=Couleur)

        # label in fig
        xlim = plt.xlim()
        # Pretilly place the label using labelShift
        shift = (xlim[1]-xlim[0])/15
        # center of gravity of the cluster of EP and nu0
        G = np.append(EP_loc, nu0).mean()

        # plotting loop for EP
        EPlabel = ''
        for i, foundEP in enumerate(EP_loc):
            cshift = labelShift(G, foundEP, shift)
            if NEP > 1:
                # add label only if several EP, label start at 1 !
                EPlabel = ''.join(('_', str(i+1)))
            plt.text(foundEP.real + cshift.real, foundEP.imag + cshift.imag,
                     '$'+variable+EPlabel+'^*$', fontsize=14, color=Couleur,
                     horizontalalignment='center', verticalalignment='center')

        # nu0 label
        cshift = labelShift(G, nu0, shift)
        plt.text(nu0.real + cshift.real, nu0.imag + cshift.imag, '$' + variable + '_0$',
                 fontsize=14, color=Couleur,
                 horizontalalignment='center', verticalalignment='center')

        plt.axis('equal')
        plt.xlabel(r'$\mathrm{Re}\,' + variable + r' $')
        plt.ylabel(r'$\mathrm{Im}\,' + variable + r' $')

        # plt.savefig('roots.pdf', format='pdf', dpi=300)
        plt.show()
