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
Select the good polynomial module (fast fortran full python fallback version).
Neither `_fpolyval_full_py.py`, nor `_fpolyval.cpython-xxx.so`
are supposed to be imported directly. One must use the `fpoly` module.

See README for installation details.
"""
try:
    # Fortran version
    from ._fpoly import polyvalnd
except ImportError:
    # Python fallback version
    from ._fpolyval_full_py import polyvalnd

# Add the imported function to doctest
__test__ = {'polyvalnd': """
            >>> import doctest
            >>> doctest.run_docstring_examples(polyvalnd, globals())
            """}

# Usefull for import * and to add it to doctring for pdoc
__all__ = ['polyvalnd']
