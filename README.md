EasterEig
=========
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![CI-Ubuntu](https://github.com/nennigb/EasterEig/actions/workflows/ci-ubuntu.yml/badge.svg)](https://github.com/nennigb/EasterEig/actions/workflows/ci-ubuntu.yml) [![pypi release](https://img.shields.io/pypi/v/eastereig.svg)](https://pypi.org/project/eastereig/)



Consider a parametric eigenvalue problem depending on a parameter \(\nu\). This arises for instance in

  - waveguides, where the _wavenumber_ (eigenvalue) depends on the frequency (parameter)
  - waveguides with absorbing materials on the wall, where _modal attenuation_ (eigenvalue imaginary part) depends on the liner properties like impedance, density (parameter)
  - structural dynamics with a randomly varying parameter, where the _resonances frequencies_ (eigenvalue) depend on the parameter
  - ...

Exceptional points (EP) of non-Hermitian systems correspond to particular values of the parameter leading to defective eigenvalue.
At EP, both eigenvalues and eigenvectors are merging. 

The aim of this package is to **locate** exceptional points and to **reconstruct** the eigenvalue loci. The theoretical part of this work is described in [1], as for the location of _exceptional points_ and illustrated in [2] for eigenvalues reconstruction in structural dynamics.

The method requires the computation of successive derivatives of two selected eigenvalues with respect to the parameter so that, after recombination, regular functions can be constructed. This algebraic manipulation enables
 * exceptional points (EP) localization, using standard root-finding algorithms; 
 * computation of the associated Puiseux series up to an arbitrary order.
  
This representation, which is associated with the topological structure of Riemann surfaces, allows to efficiently approximate the selected pair in a certain neighborhood of the EP.

To use this package :

  1. an access to the **operator derivative** must be possible
  2. the eigenvalue problem must be recast into
  		\[ \mathbf{L} (\lambda(\nu), \nu) \mathbf{x} (\nu) =\mathbf{0}  \]

The matrices of discrete operators can be either of numpy type for _full_, scipy type for _sparse_ or petsc mpiaij type for _sparse parallel_ matrices.

If eastereig is useful for your research, please cite the following references. If you have some questions, suggestions or find some bugs, report them as issues [here](https://github.com/nennigb/EasterEig/issues).

References
----------

   [1] B. Nennig and E. Perrey-Debain. A high order continuation method to locate exceptional points and to compute Puiseux series with applications to acoustic waveguides. J. Comp. Phys., 109425, (2020). [[doi](https://dx.doi.org/10.1016/j.jcp.2020.109425)]; [[open access](https://arxiv.org/abs/1909.11579)]

   [2] M. Ghienne and B. Nennig. Beyond the limitations of perturbation methods for real random eigenvalue problems using Exceptional Points and analytic continuation. Journal of Sound and vibration, (2020). [[doi](https://doi.org/10.1016/j.jsv.2020.115398)]; [[open access](https://hal.archives-ouvertes.fr/hal-02536849)]
       

Basic workflow and class hierarchy
----------------------------------

`eastereig` provides several top level classes:

  1. **OP class**, defines operators of your problem
  2. **Eig class**, handles eigenvalues, their derivatives and reconstruction
  3. **EP class**, combines Eig object to locate EP and compute Puiseux series
  4. **Loci class**, stores numerical value of eigenvalues loci and allows easy Riemann surface plotting

Dependencies
-------------

`eastereig` is based on numpy (full) and scipy (sparse) for most internal computation and can handle _large_ parallel sparse matrices thanks to **optional** import of [petsc4py](https://petsc4py.readthedocs.io/en/stable/install.html) (and mumps), 
[slepc4py](https://slepc4py.readthedocs.io/en/stable/install.html) and
and [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html). As non-hermitian problems involve complex-valued eigenvalues, computations are realized with complex arithmetic and the **complex petsc version** is expected.
Tested for python >= 3.5

> **Remarks :**
> To run an example with petsc (parallel), you need to run python with `mpirun` (or `mpiexec`). For instance, to run a program with 2 proc
> `mpirun -n 2 python myprog.py`

Riemann surface can also be plotted using the `Loci` class either with `matplotlib` or with [`pyvista`](https://github.com/pyvista/pyvista) (optional).

Install 
--------

You'll need : 
* python (tested for v >= 3.5);
* python packages: numpy, setuptools, wheel
* pip (optional).
* fortran compiler (optional)
Note that on ubuntu, you will need to use `pip3` instead of `pip` and `python3` instead of `python`. Please see the steps given in the continous integration script [workflows](.github/workflows/ci-ubuntu.yml).


By default, the fortan evaluation of multivariate polynomial is desactivated. To enable it, set the environnement variable: `EASTEREIG_USE_FPOLY=True`. On ubuntu like system, run
```console
export EASTEREIG_USE_FPOLY=True
```

### Using pip (preferred)
Consider using `pip` over custom script (rationale [here](https://pip.pypa.io/en/stable/reference/pip_install/)). 

You can install `eastereig` either from pypi (main releases only):
```
pip install eastereig [--user]
``` 
or from github:
```
pip install path/to/EeasterEig-version.tar.gz [--user]
```
or in _editable_ mode if you want to modify the sources
```
pip install -e path/to/EeasterEig
```
> The version of the required libraries specified in `install_requires` field from `setup.py` are given to ensure the backward compatibility up to python 3.5. A more recent version of these libraries can be safely used for recent python version. 

### Using python setuptools
Go to root folder.
and run:
```
python setup.py install [--user]
```

To get the lastest updates (dev relases), run: 
```
python setup.py develop [--user]
```

Running tests
-------------
Tests are handled with doctest. 

To execute the full test suite, run :
```
python -m eastereig
```

Documentation
--------------

## Generate documentation 
Run: 
```
pdoc3 --html --force --config latex_math=True  eastereig
```
N.B: The doctring are compatible with several Auto-generate API documentation, like pdoc.
This notably allows to see latex includes.

## Generate class diagram
Run: 
```
pyreverse -s0 eastereig -m yes -f ALL
dot -Tsvg classes.dot -o classes.svg
```
N.B: Class diagram generation is done using `pyreverse` (installed with pylint and spyder).

## Generate documentation and class diagram
Both aspects are included in the `makedoc.py' script. So, just run :
```
python ./makedoc.py
```

Getting started
---------------

Several working examples are available in `./examples/` folder
  
  1. Acoustic waveguide with an impedance boundary condition (with the different supported linear libraries)
  2. 3-dof toy model of a structure with one random parameter (with numpy)

To get started, the first step is to define your problem. Basically it means to link the discrete operators (matrices) and their derivatives to the `eastereig` OP class.
The problem has to be recast in the following form:

\( \left[ \underbrace{1}_{f_0(\lambda)=1} \mathbf{K}_0(\nu) + \underbrace{\lambda(\nu)}_{f_1(\lambda)=\lambda} \mathbf{K}_1(\nu) + \underbrace{\lambda(\nu)^2}_{f_2(\lambda)} \mathbf{K}_2(\nu) \right] \mathbf{x} =  \mathbf{0} \).

Matrices are then stacked in the variable `K`
```python
K = [K0, K1, K2].
```
**The functions** that return the derivatives with respect to \(\nu\) of each matrices have to be put in `dK`. The prototype of this function is fixed (the parameter n corresponds to the derivative order) to ensure automatic computation of the operator derivatives.
```python
dK = [dK0, dK1, dK2].
```
Finally, **the functions** that returns derivatives with respect to \( \lambda\) are stored in 'flda'
```python
flda = [None, ee.lda_func.Lda, ee.lda_func.Lda2].
```
Basic linear and quadratic dependency are defined in the module `lda_func`. Others dependencies can be easily implemented; provided that the appropriate eigenvalue solver is also implemented). The `None` keyword is used when there is no dependency to the eigenvalue, e. g. \(\mathbf{K}_0\).

This formulation is used to automatically compute (i) the successive derivatives of the operator and (ii) the RHS associated to the bordered matrix.

These variables are defined by creating **a subclass** that inherits from the eastereig **OP class**. For example, considering a generalized eigenvalue problem \( \left[\mathbf{K}_0(\nu) + \lambda \mathbf{K}_1(\nu) \right] \mathbf{x} =  \mathbf{0} \) :

```python
import eastereig as ee

class MyOP(ee.OP):
    """Create a subclass of the OP class to describe your problem."""

    def __init__(self):
        """Initialize the problem."""
        # Initialize OP interface
        self.setnu0(z)

        # Mandatory -----------------------------------------------------------
        self._lib = 'scipysp'  # 'numpy' or 'petsc'
        # Create the operator matrices
        self.K = self.CreateMyMatrix()
        # Define the list of function to compute the derivatives of each operator matrix (assume 2 here)
        self.dK = [self.dmat0, self.dmat1]
        # Define the list of functions to set the eigenvalue dependency of each operator matrix
        self.flda = [None, ee.lda_func.Lda]
        # ---------------------------------------------------------------------

    def CreateMyMatrices(self, ...):
        """Create my matrices and return a list."""
        ...
        return list_of_Ki

    def dmat0(self, n):
        """Return the matrix derivative with respect to nu.

        N.B. : The prototype of this function is fixed, the n parameter
        stands for the derivative order. If the derivative is null,
        the function returns the value 0.
        """
        ...
        return dM0

    def dmat1(self, n):
        """Return the matrix derivative with respect to nu.

        N.B. : The prototype of this function is fixed, the n parameter
        stands for the derivative order. If the derivative is null,
        the function returns the value 0.
        """
        ...
        return dM1
```

How to contribute ?
-------------------

If you want to contribute to `eastereig`, your are welcomed! Don't hesitate to
  - report bugs, installation problems or ask questions on [issues](https://github.com/nennigb/EasterEig/issues);
  - propose some enhancements in the code or in documentation through **pull requests** (PR);
  - suggest or report some possible new usages and why not start a scientific collaboration ;-)
  - ...
  
To ensure code homogeneity among contributors, we use a source-code analyzer (eg. pylint). 
Before submitting a PR, run the tests suite.


License
-------
This file is part of eastereig, a library to locate exceptional points and to reconstruct eigenvalues loci.

Eastereig is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Eastereig is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Eastereig.  If not, see <https://www.gnu.org/licenses/>.
