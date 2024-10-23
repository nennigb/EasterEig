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
##Define the Loci class, that provides Riemann Surface plotting capability.

The default back end is `matplotlib` but `pyvista` can be used if available.

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
<...
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

import itertools as it


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

    def __init__(self, LAMBDA=None, NU=None):
        """ Initialisation with numpy nd array.
        """
        # TODO add check on LAMDA to be consitent with plotting constaint
        # TODO asarray ?
        self.LAMBDA = LAMBDA
        self.NU = NU

    def __repr__(self):
        """ Define the object representation.
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
        npzfile = np.load(LAMBDAfile)  # where npzfile is a dict
        LAMBDA = npzfile['LAMBDA']
        NU = npzfile['NU']
        L = Loci(LAMBDA, NU)
        return L

    def export(self, LAMBDAfile):
        """ Convert and export to numpy file format.

        Parameters
        ----------
        LAMBDAfile: string
            file to save the data
        """
        np.savez(LAMBDAfile, LAMBDA=self.LAMBDA, NU=self.NU)

    # %% Using matlab
    def plotRiemannMatlab(self, part, eng, n=3, label=[r'$\nu$', r'$\lambda']):
        """ Plot Riemman surfaces with matlab [optional].

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
            from ME4pyUtils import np2mlarray
        except ModuleNotFoundError:
            raise ModuleNotFoundError('This optional method requires ME4pyUtils.')

        # extract the real or imaginary part
        if part == 'Re':
            LAMBDAc_m = np2mlarray(self.LAMBDA.real[:, :, 0:n])
        elif part == 'Im':
            LAMBDAc_m = np2mlarray(1j*self.LAMBDA.imag[:, :, 0:n])
        else:
            raise NameError("Wrong argument, part must be 'Re' or 'Im'")

        NU_real = np2mlarray(self.NU.real)
        NU_imag = np2mlarray(self.NU.imag)
        _ = eng.PlotRiemann(LAMBDAc_m, NU_real, NU_imag, part, n, label)

    # %% Using matplotlib
    def plotRiemann(self, Type='Re', N=2, EP_loc=None, Title='empty',
                    Couleur='k', variable='\\nu', fig=-2, nooutput=False):
        """ Plot Riemman surfaces of the selected eigenvalues.

        The real or imaginary part of the values to be plotted are sorted and
        the delaunay tessalation are used to draw possibility dicontinuous
        surfaces.

        Parameters
        -----------
        Type : {'Re','Im'}, default: 'Re'
            Specify which part of the complex parameter is plotted.
        N : int, default: 2
            Maximum number of eigenvalues to be plotted.
        EP_loc : (N-1,) array_like
            Values of the N-1 EP between each consecutive eigenvalues
            When specifyed, the EP position is plotted.
        Title : str, default: 'empty'
            Figure title.
        Couleur : str, default: 'k'
            Annotation color
        variable : str, default: '\\nu'
            Variable name for axis labels.
        fig : int, default: -2
            Figure numerotation.
        nooutput : bool
            Remove ploting (usefull for tests).

        Returns
        -------
        fig: fig
            The pyplot fig.
        ax: axes
            The pyplot axes for futher plot.

        """
        # TODO : Add plotting option to plot the intersection of the Riemann surface
        # with a given plane (for instance with the real plane to illustrate real
        # eigenvalue loci)

        lda = np.asarray(self.LAMBDA)
        nur = self.NU.real
        nui = self.NU.imag

        if np.shape(lda)[:2] != np.shape(nur) != np.shape(nui):
            print('Warning : lda, nur and nui shapes are not consistent')
            return

        # Generate ldaplot with reordered data from lda
        if Type == 'Re':
            indx = np.argsort(lda.real, axis=2)
        elif Type == 'Im':
            indx = np.argsort(lda.imag, axis=2)
        elif Type == 'Im_abs':
            indx = np.argsort(np.abs(lda.imag), axis=2)
        elif Type == 'Re_abs':
            indx = np.argsort(np.abs(lda.real), axis=2)
        ldaplot = np.take_along_axis(lda, indx, axis=2)

        # u, v = np.meshgrid(Pr, nui)
        u, v = nur.flatten(), nui.flatten()
        # Triangulate parameter space using Delaunay triangulation
        tri = mtri.Triangulation(u, v)

        # Plot
        Fig = plt.figure(num=fig)
        ax = Fig.add_subplot(111, projection='3d')
        for mode in range(min(len(lda), N)):
            if Type == 'Re':
                ax.plot_trisurf(u, v, ldaplot[:, :, mode].flatten().real,
                                triangles=tri.triangles, cmap=plt.cm.Spectral)
            elif Type == 'Im':
                ax.plot_trisurf(u, v, ldaplot[:, :, mode].flatten().imag,
                                triangles=tri.triangles, cmap=plt.cm.Spectral)

        # Add EP plot
        if EP_loc is not None:
            Xlim = ax.get_xlim()
            YLim = ax.get_ylim()
            Zlim = ax.get_zlim()
            for i in range(len(EP_loc)):
                ax.plot([EP_loc[i].real, EP_loc[i].real], [EP_loc[i].imag, EP_loc[i].imag],
                        Zlim, linestyle='--', color=Couleur, linewidth=0.5)
                shift = ((Xlim[1]-Xlim[0])/40.)
                ax.text(EP_loc[i].real + shift, EP_loc[i].imag + shift,
                        Zlim[0] + 2*shift, '$EP_%i$' % i, color=Couleur)

        # Fancy plot
        if Title != 'empty':
            # Fig.canvas.set_window_title(Title)  # Depreciated since mpl 3.4
            Fig.canvas.manager.set_window_title(Title)
        ax.set_xlabel(r'$\mathrm{Re}\,' + variable + r' $')
        ax.set_ylabel(r'$\mathrm{Im}\,' + variable + r' $')
        if Type == 'Re':
            ax.set_zlabel(r'$\mathrm{Re} \lambda $')
        else:
            ax.set_zlabel(r'$\mathrm{Im} \lambda $')
        if not(nooutput):
            plt.show()
        return Fig, ax

    def plotDiscriminant(self, index=None, variable='\\nu', scale='log',
                         normalize=True, fig=None, clip=(0.01, 0.99),
                         cmap=plt.cm.turbo, gradient=False):
        r"""Plot of the modulus of the discriminant of the 'partial'
        characteristic polynomial.

        This analytic function vanishes for all multiple roots and is
        defined as
        $$H(\nu) = \prod_{i<j} (\lambda_i(\nu)-\lambda_j(\nu))^2.$$
        The eigenvalue are picked in `index` if given, else all eigenvalues
        present in the Loci instance are used.
        Depending of the dynamic of the eigenvalue, this method can yields
        to 'nan'. To limit that, complex256 are used locally.

        Parameters
        -----------
        index : array_like
            The index of the eigenvalue to combine. The default is `None` (all).
        variable : str
            Variable name for axis labels. The default: '\\nu'.
        scale : string
            scaling of the discriminant ('lin', 'log'). The default is 'log'.
        normalize : bool
            if true normalize lda using goemetric average. The default is True.
        fig : int or string
            Figure label for superposition. The default: None, create a new figure.
        clip : tuple
            `clip` represents the min and max quantiles to set the color
            limits of the current image. Using `clip` avoid that isolate
            extreme values disturbed the image.
        cmap : cmap
            Change the default colormap. The default is the high
            contrast `turbo` corlormap.
        gradient : bool, optional
            Plot also the magnitude of the gradient on another figure.

        Returns
        -------
        fig : fig
            The pyplot fig
        ax : axes
            The pyplot axes for futher plot
        ax_g : axes, optional
            The pyplot axes of the gradient if `gradient=True`, else `None`.
        """

        lda = np.asarray(self.LAMBDA)
        nur = self.NU.real
        nui = self.NU.imag

        if np.shape(lda)[:2] != np.shape(nur) != np.shape(nui):
            print('Warning : lda, nur and nui shapes are not consistent')
            return

        # Normalize lda to limit Overflow
        if normalize:
            lda_min = np.abs(lda).min()
            lda_max = np.abs(lda).max()
            lda_norm = np.sqrt(lda_min*lda_max)
            # lda_norm = (lda_max - lda_min)/2
            lda = lda / lda_norm

        if index is None:
            index = np.arange(0, lda.shape[2])
        # initialize the product to 1.
        P = np.ones_like(nur, dtype=np.complex256)  # possible to use quad complex
        # get all combinaison of eigenvalue and compute dh(i pair)
        for i in it.combinations(index, 2):
            P *= ((lda[:, :, i[0]] - lda[:, :, i[1]])**2).astype(np.complex256)
        if np.isnan(P).any():
            print("Warning : 'P' contains 'nan' in 'plotDiscriminant'.",
                  " Try to plot...")
        # try to plot
        Fig = plt.figure(num=fig)
        ax = Fig.add_subplot(111)
        if scale == 'lin':
            im = plt.pcolormesh(nur, nui, np.abs(P), zorder=-1, cmap=cmap)
        elif scale == 'log':
            field = np.log10(np.abs(P))
            im = plt.pcolormesh(nur, nui, field, zorder=-1, cmap=cmap)
            stats = (field.min(), field.max(), field.mean())
            print('> stats:', stats)
        # im = plt.pcolormesh(nur, nui, P.imag, zorder=-1)
        plt.xlabel(r'$\mathrm{Re}\,' + variable + r' $')
        plt.ylabel(r'$\mathrm{Im}\,' + variable + r' $')
        plt.colorbar()
        im.set_clim(np.quantile(field.ravel(), clip[0]),
                    np.quantile(field.ravel(), clip[1]))
        if gradient:
            df = np.gradient(field)
            m = np.sqrt(df[0]**2 + df[1]**2)
            fig_g, ax_g = plt.subplots()
            im_g = ax_g.pcolormesh(nur, nui, m, zorder=-1, cmap=cmap)
            ax_g.set_xlabel(r'$\mathrm{Re}\,' + variable + r' $')
            ax_g.set_ylabel(r'$\mathrm{Im}\,' + variable + r' $')
        else:
            ax_g = None
        return Fig, ax, ax_g



# %% Using pyvista
    def plotRiemannPyvista(self, Type='Re', N=2, Title='empty',
                           variable='\\nu', lda_list=None, qt=True, normalize=1.):
        """ Plot Riemman surfaces of the selected eigenvalues using pyvista.

        The real or imaginary part of the values to be plotted are sorted and
        the delaunay tessalation are used to draw possibility dicontinuous
        surfaces. The color is given by the other part. eg if `Type='Re'`,
        the color is given by the imaginary part. It allows to see an EP,
        since both height and color should be the same.

        Some key can be used to interact with the plot:
            'p' allows to pick points (may break if zscale is modified)
            't' allows to toogle to 'points' representation
            'w' allows to toogle to 'wireframe' representation
            's' allows to toogle to 'surface' representation
            'q' to quit.

        If the differents surfaces are needed, they can be access to the
        Plotter properties `_datasets`.

        Parameters
        -----------
        Type : {'Re','Im'}, default: 'Re'
            Specify which part of the complex parameter is plotted.
        N : int, default: 2
            Maximum number of eigenvalues to be plotted.
        Title : str, default: 'empty'
            Figure title.
        variable : str, default: '\\nu'
            Variable name for axis labels.
        qt : bool
            Flag to swtich between `pyvista` and `pyvistaqt`. It allows to plot
            in background using another thread.
        normalize: float, optional
            Normalize the eigenvalue using the `normalize` scaling constant. The
            default is 1.

        Returns
        -------
        pv: Plotter
            The pyvista plotter object.
        picked: list
            The list of the coordinates of the picked points.
        """
        try:
            import pyvista as pv
            # pyvistaqt allows to plot in background
            if qt:
                try:
                    from pyvistaqt import BackgroundPlotter as Plotter
                except ModuleNotFoundError:
                    print('The module `pyvistaqt` is not installed. Set `qt=False`.')
            else:
                Plotter = pv.Plotter
        except ModuleNotFoundError:
            print('The module `pyvista` is not installed.')
            return

        # Default size of the check box
        Startpos = 12
        size = 50

        # Define a Class to hide or add an actor
        class SetVisibilityCallback:
            """Helper callback to keep a reference to the actor being modified."""
            def __init__(self, actor):
                self.actor = actor

            def __call__(self, state):
                # self.actor.SetVisibility(state)
                if state:
                    p.add_actor(self.actor)
                else:
                    p.remove_actor(self.actor)
                p.update_bounds_axes()


        # Recast data for plotting
        if lda_list is not None:
            lda = np.asarray(self.LAMBDA[:, :, lda_list])
            if N> len(lda_list): N=len(lda_list)
        else:
            lda = np.asarray(self.LAMBDA)
        nur = self.NU.real
        nui = self.NU.imag

        if np.shape(lda)[:2] != np.shape(nur) != np.shape(nui):
            print('Warning : lda, nur and nui shapes are not consistent')
            return

        # Generate ldaplot with reordered data from lda
        if Type == 'Re':
            indx = np.argsort(lda.real, axis=2)
        elif Type == 'Im':
            indx = np.argsort(lda.imag, axis=2)
        elif Type == 'Im_abs':
            indx = np.argsort(np.abs(lda.imag), axis=2)
        elif Type == 'Re_abs':
            indx = np.argsort(np.abs(lda.real), axis=2)
        ldaplot = np.take_along_axis(lda, indx, axis=2)

        # u, v = np.meshgrid(Pr, nui)
        u, v = nur.flatten(), nui.flatten()

        # Create all the surfaces
        surfs = []
        col_name = {'Re': 'Im λ', 'Im': 'Re λ'}
        for mode in range(min(len(lda), N)):
            if Type == 'Re':
                points = np.column_stack([u, v,  normalize * ldaplot[:, :, mode].ravel().real])
                col = ldaplot[:, :, mode].ravel().imag
            elif Type == 'Im':
                points = np.column_stack([u, v,  normalize * ldaplot[:, :, mode].ravel().imag])
                col = ldaplot[:, :, mode].ravel().real
            cloud = pv.PolyData(points)
            cloud[col_name[Type]] = col
            # Triangulate parameter space using Delaunay triangulation
            surfs.append(cloud.delaunay_2d())

        # Add them to the plotter object
        p = Plotter(title='Riemann surface')
        actors = []
        for surf in surfs:
            actors.append(p.add_mesh(surf, show_edges=True, pickable=True,
                                     render_points_as_spheres=True, point_size=10.0))

        # Add a check box for all actors
        for actor in actors:
            callback = SetVisibilityCallback(actor)
            p.add_checkbox_button_widget(callback, value=True,
                                         position=(5.0, Startpos), size=size,
                                         border_size=1,
                                         color_on='red',
                                         color_off='grey',
                                         background_color='grey')
            Startpos = Startpos + size + (size // 10)

        # Define the scale slider
        def scale(zscale):
            p.set_scale(zscale=zscale)
        # FIXME set scale break the picking https://github.com/pyvista/pyvista/issues/1335
        slider = p.add_slider_widget(scale, [0.1, 10], value=1, title='z scaling')

        # Define axis layout
        # FIXME still trouble with tex and utf8 chars...
        xlabel = u'Re ν'
        ylabel = 'Im \u03bd'
        if Type == 'Re':
            zlabel = 'Re λ'
        else:
            zlabel = 'Im λ'
        # axes_actor.SetXAxisLabelText(xlabel)
        p.show_grid(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

        # Define picker
        # Add a picker label style
        dargs = dict(name='labels', font_size=24)
        picked = []
        def picker_callback(pid):
            """Plot the picked node label."""
            label = str(np.round(pid, decimals=2))
            print('> picked point at :', pid)
            p.add_point_labels(pid, [label], **dargs)
            picked.append(pid)

        p.enable_point_picking(callback=picker_callback, show_message=True,
                               color='pink', point_size=15,
                               show_point=True)

        # TODO add a clip-box widget
        # p.add_mesh_clip_box(surfs[0], color='white')

        # Add a key to switch to 'Points' representation
        def toogle_to_point():
            """Toogle all Riemann surface to points reprensentation."""
            for actor in actors:
                prop = actor.GetProperty()
                prop.SetRepresentationToPoints()
            p.update()
        p.add_key_event('t', toogle_to_point)
        p.show()  # auto_close=True,

        return p, picked
