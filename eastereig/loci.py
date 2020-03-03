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
##Define the Loci class, that provides Riemann Surface plotting capability

Example
--------
Plot Riemann surface 
>>> nu_real, nu_imag = np.meshgrid(np.linspace(-0.2,0.2,31),np.linspace(0,0.7,31))
>>> NU= 1.*(nu_real+1j*nu_imag) 
>>> LDA =  np.zeros((NU.shape[0],NU.shape[1],2), dtype=np.complex128)                
>>> LDA[:,:,0]=NU**2
>>> LDA[:,:,1]=-NU**2                
>>> RS = Loci(LDA,NU)  
>>> RS.plotRiemann('Re',nooutput=True)[1] # doctest: +ELLIPSIS
<matplotlib.axes._subplots.Axes3DSubplot object at ...
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

    
    

class Loci:
    """
    Provides Riemann Surface plotting capability

    Attributes
    ----------
    NU: np.array
        the 2D map of the parameter
    LAMBDA: (np.shape(NU),n_eig) list
            where n_eig is the number of eigenvalue
            
    """
    def __init__(self,LAMBDA=None,NU=None):
        """
        Initialisation with numpy nd array
        """
        #todo add check on LAMDA to be consitent with plotting constaint
        self.LAMBDA = LAMBDA
        self.NU = NU

    def __repr__(self):
       """ Define the object representation        
       """
       return "Instance of Loci class. Contains {}x{} nu map.".format(*self.NU.shape)


    @staticmethod
    def reloadLoci(LAMBDAfile):
        """ Load from save npy file and return a Loci instance

        Parameters
        ----------
        LAMBDAfile: string
            the name of the file to reload
        """
        npzfile = np.load(LAMBDAfile) #where npzfile is a dict
        LAMBDA = npzfile['LAMBDA']
        NU = npzfile['NU']
        L = Loci(LAMBDA,NU)
        return L
        
    def export(self, LAMBDAfile):
        """ Convert and export to numpy file format.

        Parameters
        ----------
        LAMBDAfile: string
            file to save the data
        """
        np.savez(LAMBDAfile, LAMBDA=self.LAMBDA, NU=self.NU)

    
    def plotRiemannMatlab(self,part,eng,n=3,label=['$\nu$','$\lambda']):
        """
        Plot Riemman surfaces with matlab [optional]
        
        Remarks
        -------
        This optional method requires ME4pyUtils and matlab engine to plot 
        Riemman with matlab.
        ```
        from ME4pyUtils import np2mlarray
        ```
        `PlotRiemann` (full python) must be prefered. 
        
        Parameters
        -----------
        part: {'Re','Im'}
            Specify which part of the complex parameter is plotted
        eng: matlab engine instance
            matlab engine should running
        n: int, default: 3
            Maximum number of eigenvalues to be plotted
        label : list
            list of (x,y) axis and z axis latex label
        """
        try:
            from ME4pyUtils import np2mlarray, mlarray2np
        except:
            raise NameError('This optional method requires ME4pyUtils')

        # extract the real or imaginary part
        if part=='Re':
            LAMBDAc_m = np2mlarray(self.LAMBDA.real[:,:,0:n])
        elif part =='Im':
            LAMBDAc_m = np2mlarray(1j*self.LAMBDA.imag[:,:,0:n])
        else:
            raise NameError("Wrong argument, part must be 'Re' or 'Im'")
        
        NU_real=np2mlarray(self.NU.real)
        NU_imag=np2mlarray(self.NU.imag)
        out = eng.PlotRiemann(LAMBDAc_m,NU_real,NU_imag,part,n,label)

    def plotRiemann(self,Type='Re',N=2,EP_loc = None,Title='empty',Couleur='k',variable='\\nu',fig=-2,nooutput=False):
        """
        Plot Riemman surfaces of the selected eigenvalues
        
        The real or imaginary part of the values to be plotted are sorted and the delaunay tessalation 
        are used to draw possibility dicontinuous surfaces

        Parameters
        -----------
        Type: {'Re','Im'}, default: 'Re'
            Specify which part of the complex parameter is plotted            
        N: int, default: 2
            Maximum number of eigenvalues to be plotted    
        EP_loc: (N-1,) array_like 
            Values of the N-1 EP between each consecutive eigenvalues
            When specifyed, the EP position is plotted        
        Title: str, default: 'empty'
            Figure title            
        Couleur: str, default: 'k'
            Annotation color            
        variable: str, default: '\\nu'
            Variable name for axis labels            
        fig: int, default: -2
            Figure numerotation        
        nooutput : bool
            remove ploting (usefull for tests)

        Returns
        -------
        fig: fig 
            the pyplot fig 
        ax: axes
            the pyplot axes for futher plot
        
        """
        #TODO : Add plotting option to plot the intersection of the Riemann surface
        #with a given plane (for instance with the real plane to illustrate real
        #eigenvalue loci)

        lda=np.asarray(self.LAMBDA)
        nur = self.NU.real
        nui = self.NU.imag
        
        if np.shape(lda)[:2] != np.shape(nur) != np.shape(nui):
            print('Warning : lda, nur and nui shapes are not consistent')
            return
        
        ### Generate ldaplot with reordered data from lda
        if Type == 'Re':
            indx = np.argsort(lda.real,axis=2)
        elif Type == 'Im':
            indx = np.argsort(lda.imag,axis=2)
        ldaplot = np.take_along_axis(lda, indx, axis=2)

        #u, v = np.meshgrid(Pr, nui)
        u, v = nur.flatten(), nui.flatten()
        tri = mtri.Triangulation(u,v) # Triangulate parameter space using Delaunay triangulation
    
        ### Plot
        #Fig = plt.figure(num=fig,figsize=plt.figaspect(.5))
        #ax = Fig.add_subplot(1, 2, 1, projection='3d')
        Fig = plt.figure(num=fig)
        ax = Fig.add_subplot(111, projection='3d')
        for mode in range(min(len(lda),N)):
            if Type == 'Re':
                ax.plot_trisurf(u,v, ldaplot[:,:,mode].flatten().real, triangles=tri.triangles, cmap=plt.cm.Spectral)
            elif Type == 'Im':
                ax.plot_trisurf(u,v, ldaplot[:,:,mode].flatten().imag, triangles=tri.triangles, cmap=plt.cm.Spectral)
        
        ### Add EP plot
        if EP_loc != None:
            Xlim = ax.get_xlim()
            YLim = ax.get_ylim()
            Zlim = ax.get_zlim()
            for i in range(len(EP_loc)):
                ax.plot([EP_loc[i].real, EP_loc[i].real],[EP_loc[i].imag, EP_loc[i].imag], Zlim,linestyle='--',color=Couleur, linewidth=0.5)
                shift = ((Xlim[1]-Xlim[0])/40.);
                ax.text(EP_loc[i].real + shift, EP_loc[i].imag + shift, Zlim[0] + 2*shift, '$EP_%i$'%i,color=Couleur)
        
        ### Fancy plot
        if Title!='empty':
            Fig.canvas.set_window_title(Title)
        ax.set_xlabel(r'$\mathrm{Re}\,'+ variable +r' $')
        ax.set_ylabel(r'$\mathrm{Im}\,'+ variable +r' $')
        if Type == 'Re':
            ax.set_zlabel(r'$\mathrm{Re} \lambda $')
        else:
            ax.set_zlabel(r'$\mathrm{Im} \lambda $')
        if not(nooutput):
            plt.show()
        return Fig,ax
