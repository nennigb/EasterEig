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
# Test suite runner

Run `eastereig` run the test suite using `doctest` and `unittest` framework.

Example
-------
```console
python3 -m eastereig
```
"""
import unittest
import doctest
import sys
import numpy as np
from eastereig import _petscHere
# Import the file containing the doctest
from eastereig.examples import WGimpedance_numpy
from eastereig.examples import WGimpedance_scipysp
from eastereig.examples import ThreeDoF
from eastereig.examples import toy_3dof_2params
from eastereig.examples import WGadmitance_numpy_mv
from eastereig.examples import WGadmitance_scipy_mv
from eastereig import utils
from eastereig import loci
from eastereig import ep
from eastereig import lda_func
from eastereig import eigSolvers
from eastereig import fpoly
from eastereig import charpol

# Numpy 2.0 change default printing options making doctest failing.
# https://numpy.org/neps/nep-0051-scalar-representation.html
# Use legacy mode for testing
if np.lib.NumpyVersion(np.__version__) >= '2.0.0b1':
    np.set_printoptions(legacy="1.25")

if _petscHere:
    from eastereig.examples import WGimpedance_petsc
    from eastereig.examples import WGadmitance_petsc_mv

# Explicitely list modules with doctest
mod_list = [lda_func, utils, loci, ep, eigSolvers, fpoly, charpol,
            WGimpedance_numpy, WGimpedance_scipysp, ThreeDoF,
            toy_3dof_2params, WGadmitance_numpy_mv, WGadmitance_scipy_mv]
if _petscHere:
    petsc_list = [WGimpedance_petsc, WGadmitance_petsc_mv]
    mod_list.extend(petsc_list)

if __name__ == '__main__':
    import os
    tests_dir = os.path.join(os.path.dirname(__file__), 'tests')
    print('> Running tests...')
    Stats = []
    # Create test suite for unittest and doctest
    suite = unittest.TestLoader().discover(start_dir=tests_dir, pattern='test*.py')
    # Add doctest from all modules of mod_list
    for mod in mod_list:
        suite.addTest(doctest.DocTestSuite(mod,
                                            optionflags=(doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    # Summary, with petsc output is sometime hard to read
    print("\n", "================ Testing summary ===================")
    if result.wasSuccessful():
        print("                                             Pass :-)")
        sys.exit(0)
    else:
        print("                                           Failed :-(")
        sys.exit(1)
