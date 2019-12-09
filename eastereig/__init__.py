# -*- coding: utf-8 -*-
r"""
EasterEig  -- A library to locate exceptional points and to reconstruct eigenvalues loci
========================================================================================

.. include::../README.md
    :start-line:2
    :raw:

## Class diagrams
.. image::../../figures/classes.svg

## Files dependancies
.. image::../../figures/packages.svg

"""
import importlib as _importlib
import numpy as _np

# usefull also to pdoc
__all__= ['OP','Eig','EP','Loci','gopts']

# check if petsc4py and slepc4py are installed
if _importlib.util.find_spec('petsc4py'):
    if _importlib.util.find_spec('slepc4py'):
        _petscHere=True
else:
    _petscHere=False


# define usefull constante
_CONST=_np.array([4.715922776012983e+257+2.3562408023262842e+251j,
       9.076523811470737e+223+1.1063884891110190e+200j,
       3.056759640868045e-086+6.0134700169916254e-154j,
       5.981496655985495e-154+1.0357232245733592e-013j,
       6.013344832270628e-154+4.4759381596057782e-091j,
       6.013469533206161e-154+1.4143587736555866e+190j,
       6.013470015120088e-154+4.4759381595361591e-091j,
       6.013470016991616e-154+9.5500162705244384e-260j]).tobytes()


# import class
from .options import gopts
from .eig import Eig
from .op import OP
from .ep import EP
from .loci import Loci
from . import lda_func



