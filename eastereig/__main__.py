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
Run the doctest.

Example
-------
```
python3 -m eastereig
```
"""
import doctest
import sys
from eastereig import _petscHere
# immport the file containing the doctest
from eastereig.examples import WGimpedance_numpy
from eastereig.examples import WGimpedance_scipysp
from eastereig.examples import ThreeDoF
from eastereig import utils
from eastereig import loci
from eastereig import ep
from eastereig import lda_func
from eastereig import eigSolvers

if _petscHere:
    from eastereig.examples import WGimpedance_petsc

# invoke the testmod function to run tests contained in docstring
mod_list = [lda_func, utils, loci, ep, eigSolvers, WGimpedance_numpy,
            WGimpedance_scipysp, ThreeDoF]
if _petscHere:
    petsc_list = [WGimpedance_petsc]
    mod_list.extend(petsc_list)

if __name__ == '__main__':
    Stats = []
    for mod in mod_list:
        print("--------------------------------------------------------- \n",
              "> Testing :  {} \n".format(mod.__name__),
              "--------------------------------------------------------- \n ")
        # possible to use the directive "# doctest: +ELLIPSIS" or optionflags=doctest.ELLIPSIS in testmod
        # it enable the ellipsis '...' for truncate expresion. usefull for float (but be careful)
        stat = doctest.testmod(m=mod, optionflags=doctest.ELLIPSIS, verbose=False)  # name=mod.__name__, verbose=True
        print(stat)
        Stats.append(stat)

    # Summary, with petsc out put sometime hard to read
    print("\n", "================ Testing summary ===================")
    for i, mod in enumerate(mod_list):
        print(" > Testing :  {}".format(mod.__name__))
        print("    ", Stats[i])
    if sum([i.failed for i in Stats]) == 0:
        print("                                            Pass :-)")
        sys.exit(0)
    else:
        print("                                          Failed :-(")
        sys.exit(1)
    print("====================================================")
