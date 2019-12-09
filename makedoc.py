#!/usr/bin/python3
"""
This module generateq eastereig documentation and classes diagrams.
It requieres `pdoc3` for the doc, and `pylint` and `graphviz` for the diagrams.
These modules are **optional** and are not mandatory to run `eastereig` computations.


Dependancies
------------
  - `pyreverse` included for in `pylint` (comming with spyder for instance)
  - `graphviz`

"""
# use pyreverse to generate class diagram
import subprocess
import argparse




# run command line options parser
parser = argparse.ArgumentParser(description='Generate the doc with pdoc and classes/package diagrams')
# defaut args.classes value is False (long to generate)
parser.add_argument('-c',dest='classes',action='store_true',help='re-generate diagrams', required=False)
args = parser.parse_args()

# generate classe diagrams
if args.classes:
    pyreverse = "pyreverse -s0 eastereig -m yes -f ALL".split(' ')
    dot_classes = "dot -Tsvg classes.dot -o classes.svg".split(' ')
    dot_packages= "dot -Tsvg packages.dot -o packages.svg".split(' ')
    subprocess.run(pyreverse)
    subprocess.run(dot_classes)
    subprocess.run(dot_packages)
else:
    print(" > Keep previous classes diagrams.")

# generate the doc with pdoc
pdoc="pdoc3 --html --force --config latex_math=True eastereig".split(' ')
subprocess.run(pdoc)
