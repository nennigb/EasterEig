# -*- coding: utf-8 -*-
""" 
Use this file to run the doctest when the module is run

Example
-------
```
python3 eastereig
```
"""
import doctest
from eastereig import _petscHere
# immport the file containing the doctest
from eastereig.examples import WGimpedance_numpy
from eastereig.examples import WGimpedance_scipysp
from eastereig.examples import ThreeDoF
from eastereig import utils
from eastereig import loci
from eastereig import ep
from eastereig import lda_func

if _petscHere:
    from eastereig.examples import WGimpedance_petsc

# invoke the testmod function to run tests contained in docstring
mod_list = [lda_func, utils, loci, ep, WGimpedance_numpy, WGimpedance_scipysp, ThreeDoF]
if _petscHere:
    petsc_list=[WGimpedance_petsc]
    mod_list.extend(petsc_list)


Stats=[]
for mod in mod_list:
    print( "--------------------------------------------------------- \n\
     > Testing :  {} \n\
--------------------------------------------------------- \n ".format(mod.__name__))
    # possible to use the directive "# doctest: +ELLIPSIS" or optionflags=doctest.ELLIPSIS in testmod
    # it enable the ellipsis '...' for truncate expresion. usefull for float (but be careful)
    stat=doctest.testmod(m=mod,optionflags=doctest.ELLIPSIS,verbose=False) #name=mod.__name__, verbose=True
    print(stat)
    Stats.append(stat)


#summary, with petsc out put sometime hard to read
print("\n","================ Testing summary ===================")
for i,mod in enumerate(mod_list):
    print(" > Testing :  {}".format(mod.__name__))
    print("    " ,Stats[i])
    print("\n")
if sum([i.failed for i in Stats ])==0:
    print("                                            Pass :-)")
    Pass=True
else:
    print("                                          Failed :-(")
    Pass=False
print("====================================================")



