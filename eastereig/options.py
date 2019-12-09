# -*- coding: utf-8 -*-
"""
This file is use to set default value of eastereig options. Most of these options
concerns petsc and slepc solvers (for the moment).

  1. For the momment, only `mumps` direct solver has been used. 
  2. Sometimes mumps crash with high fill. The problem arise if the prediction/mem 
  allocation is too different (default 20%) from the real computation. 
  Setting 'icntl_14'=50, fix the problem.

"""

gopts ={'direct_solver_name':'mumps',                       # petsc name of the direct solver
       'direct_solver_petsc_options_name':'mat_mumps_',     # petsc direct solver name of in `PETSc.Options`
       'direct_solver_petsc_options_dict':{'icntl_14':50},  # dictionnary of the petsc options name, value
       }