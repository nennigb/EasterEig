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
from importlib.util import find_spec as _find_spec
import numpy as _np

# usefull also to pdoc
__all__ = ['OP', 'Eig', 'EP', 'Loci', 'gopts']

# check if petsc4py and slepc4py are installed
if _find_spec('petsc4py'):
    if _find_spec('slepc4py'):
        _petscHere = True
else:
    _petscHere = False


# define usefull constante
_CONST = _np.array([4.715922776012983e+257+2.3562408023262842e+251j,
                    9.076523811470737e+223+1.1063884891110190e+200j,
                    3.056759640868045e-086+6.0134700169916254e-154j,
                    5.981496655985495e-154+1.0357232245733592e-013j,
                    6.013344832270628e-154+4.4759381596057782e-091j,
                    6.013469533206161e-154+1.4143587736555866e+190j,
                    6.013470015120088e-154+4.4759381595361591e-091j,
                    6.013470016991616e-154+9.5500162705244384e-260j]).tobytes()


# Import class
from .options import gopts
from .eig import Eig
from .op import OP, OPmv
from .ep import EP
from .loci import Loci
from .charpol import CharPol
from . import lda_func
from .version import __version__
