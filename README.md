EasterEig
------------

Consider a parametric [eigenvalue](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) problem depending on a parameter \(\nu\). This arises for instance in

  - waveguides, where the _wavenumber_ (eigenvalue) depends on the frequency (parameter)
  - waveguides with absorbing materials on the wall, where _modal attenuation_ (eigenvalue imaginary part) depends on the liner properties like impedance, density (parameter)
  - structural dynamics with a randomly varying parameter, where the _resonances frequencies_ (eigenvalue) depend on the parameter
  - ...

Exceptional points (EP) of non-Hermitian systems correspond to particular values of the parameter leading to defective eigenvalue.
At EP, both eigenvalues and eigenvectors are merging. 

The aim of this package is to **locate** exceptional points and to **reconstruct** the eigenvalue loci. The theoretical part of this work is described in [1], as for the location of _exceptional points_ and illustrated in [2] for eigenvalue reconstruction in structural dynamics.

The method requires the computation of successive derivatives of two selected eigenvalues with respect to the parameter so that, after recombination, regular functions can be constructed. This algebraic manipulation enables
 * exceptional points (EP) localization, using standard root-finding algorithms; 
 * computation of the associated Puiseux series up to an arbitrary order.
  
 This representation, which is associated with the topological structure of Riemann surfaces, allows to efficiently approximate the selected pair in EP's local neighbourhood.

To use this package :

  1. an access to the **operator derivative** must be possible
  2. the eigenvalue problem must be recast into
  		\[ \mathbf{L} (\lambda(\nu), \nu) \mathbf{x} (\nu) =\mathbf{0}  \]

Discrete operators's matrices can be of either types: numpy for _full_, scipy for _sparse_ or petsc for _sparse parallel_ matrices.

If eastereig is useful for your research, please cite the following references. If you have some questions, suggestions or find some bugs, report them as issues [here](https://github.com/nennigb/EasterEig/issues).

References
----------

.. [1] B. Nennig and E. Perrey-Debain (2019). A high order continuation method to locate exceptional points and to compute Puiseux series with applications to acoustic waveguides. arXiv preprint arXiv:1909.11579.

.. [2] M. Ghienne and B. Nennig (2019). Beyond the limitations of perturbation methods for real random eigenvalue problems using Exceptional Points and analytic continuation. submitted to journal of sound and vibrations.
       

Basic workflow and class hierarchy
--------------------------------------

`eastereig` provides several top level classes

  1. **OP class**, defines operators of your problem
  2. **Eig class**, handles eigenvalues, their derivatives and reconstruction
  3. **EP class**, combines Eig object to locate EP and compute Puiseux series
  4. **Loci class**, allows easy Riemann surface plotting

 Dependencies
--------------------------------------
`eastereig` is based on numpy (full) and scipy (sparse) for most internal computation and can handle _large_ parallel sparse matrices thanks to **optional** import of [petsc4py](https://petsc4py.readthedocs.io/en/stable/install.html) (and mumps), 
[slepc4py](https://slepc4py.readthedocs.io/en/stable/install.html) and
and [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html). As non-hermitian problems involve complex-valued eigenvalues, computations are realized with complex arithmetic and the **complex petsc version** is expected.
Tested for python 3.5

> **Remarks :**
> To run an example with petsc (parallel), you need to run python with `mpirun`. For instance, to run a program with 2 proc
> `mpirun -n 2 python3 myprog.py`

## Install (without pip)
--------------------------------------
```
python3 setup.py install [--user]
```
for frequent updates (dev) it is better to use 
```
python3 setup.py develop [--user]
```
or preferably using `pip3` as explain [here](https://pip.pypa.io/en/stable/reference/pip_install/)
```
pip3 install path/to/EeasterEig-version.tar.gz [--user]
```
or in _editable_ mode if you want to modify the sources
```
pip3 install -e path/to/EeasterEig
```

Running tests
-------------------------------------
Tests are handled with doctest. To call the test procedure, simply execute the package
```
python3 -m eastereig
```


Documentation
--------------------------------------
The doctring are compatible with several Auto-generate API documentation, like pdoc
```
pdoc3 --html --force --config latex_math=True  eastereig
```
This notably allows to see latex includes.

The class diagramm can be obtained with `pyreverse` (installed with pylint and spyder)
```
pyreverse -s0 eastereig -m yes -f ALL
dot -Tsvg classes.dot -o classes.svg

```
To generate this documentation, you can use `./makedoc.py` script.

Getting started
--------------------
Several working examples are available in `./examples/` folder
  
  1. Acoustic waveguide with an impedance boundary condition (with the different supported linear libraries)
  2. 3-dof toy model of a structure with one random parameter (with numpy)

To get started, the first step is to define your problem. Basically it means to link the discrete operators (matrices) and their derivatives to the `eastereig` OP class.
The problem has to be recast in the following form:

\( \left[ \underbrace{1}_{f_0(\lambda)=1} \mathbf{K}_0(\nu) + \underbrace{\lambda(\nu)}_{f_1(\lambda)=\lambda} \mathbf{K}_1(\nu) + \underbrace{\lambda(\nu)^2}_{f_2(\lambda)} \mathbf{K}_2(\nu) \right] \mathbf{x} =  \mathbf{0} \).

Matrices are then stacked in the variable `K`
```
K = [K0,K1,K2].
```
**The functions** that return the derivatives with respect to \(\nu\) of each matrices have to be put in `dK`. This function's prototype is fixed (parameter n corresponds derivative's order) to ensure automatic computation of the operator's derivatives.
```
dK = [dK0,dK1,dK2].
```
Finally, **the functions** that returns derivatives with respect to \( \lambda\) are stored in 'flda'
```
flda = [None,ee.lda_func.Lda,ee.lda_func.Lda2].
```
Basic linear and quadratic dependency are defined in the module `lda_func`. Others dependencies can be easily implemented; provided that the appropriate eigenvalue solver is also implemented). The `None` keyword is used when there is no dependency to the eigenvalue, e. g. \(\mathbf{K}_0\).

This formulation is used to automatically compute (i) the operator' successive derivatives and (ii) bordered matrix's RHS.  

These variables are defined by creating **a subclass** that inherits from the eastereig **OP class**. For example, considering a generalized eigenvalue problem \( \left[\mathbf{K}_0(\nu) + \lambda \mathbf{K}_1(\nu) \right] \mathbf{x} =  \mathbf{0} \) :

```
import eastereig as ee

class MyOP(ee.OP):
    """ Create a subclass of the OP class to describe your problem
    """
    def __init__(self):
        """ Initialize the problem       
        """
        
        # initialize OP interface
        self.setnu0(z)
        
        # mandatory -----------------------------------------------------------
        self._lib='scipysp' # 'numpy' or 'petsc'
        # create the operator matrices
        self.K=self.CreateMatrix()
        # define the list of function to compute each operator matrix's derivatives 
        self.dK = [self.dmat0, self.dmat1]        
        # define a function list to set the eigenvalue dependency of each operator's matrix
        self.flda = [None, ee.lda_func.Lda] 
        # ---------------------------------------------------------------------

    def CreateMyMatrices(self,...):
		""" Create my matrices and return a list
		"""
 		...   
    	return list_of_Ki
    
    def dmat0(self,n):
		""" Return the matrix's derivative with respect to nu
		N.B. : This function's prototype is fixed, n parameter
		standing for the derivative's order. If the derivative is null,
		the function returns the value 0.
		"""
		...
		return dM0
    
    def dmat1(self,n):
		""" Return the matrix's derivative with respect to nu
		N.B. : This function's prototype is fixed, n parameter
		standing for the derivative's order. If the derivative is null,
		the function returns the value 0.
		"""
		...
		return dM1
    
    
```


License
--------------------
This file is part of eastereig, a library to locate exceptional points and to reconstruct eigenvalues loci.

Eastereig is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Eastereig is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Eastereig.  If not, see <https://www.gnu.org/licenses/>.
