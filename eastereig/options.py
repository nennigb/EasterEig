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
This file is used to set default values of eastereig options. Most of these options
concerns petsc/slepc solvers or charpol.

  1. For the momment, only `mumps` direct solver has been used.
  2. Sometimes mumps crash with high fill. The problem arise if the prediction/mem
  allocation is too different (default 20%) from the real computation.
  Setting 'icntl_14'=50, fix the problem.

"""

gopts = {'direct_solver_name': 'mumps',                         # petsc name of the direct solver
         'direct_solver_petsc_options_name': 'mat_mumps_',      # petsc direct solver name of in `PETSc.Options`
         'direct_solver_petsc_options_dict': {'icntl_14': 50},  # dictionnary of the petsc options name, value
         # Number of workers in charpol multiply. Set to the number of cores for speed,
         # set to 1 for more compatibility if mixed with petsc matrices
         'max_workers_mult': 1,
         }
