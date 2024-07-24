# -*- coding: utf-8 -*-
import setuptools
import os
# Usefull to build f90 files
from numpy.distutils.core import Extension, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

# Use env var to activate fpoly, defaut is false
USE_FPOLY = os.getenv('EASTEREIG_USE_FPOLY', 'False').lower() in ('true', '1', 't')
if USE_FPOLY:
    # Define f90 module to include
    ext_fpoly = Extension(name='eastereig.fpoly._fpolyval',
                          sources=['eastereig/fpoly/fpolyval.f90'])
    ext_modules = [ext_fpoly]
    print('Use fortran fpoly submodule.', 'Set environnement variable:',
          '`EASTEREIG_USE_FPOLY=False` to activate full python version.')
else:
    ext_modules = []
    print('Use full python version. Set environnement variable:',
          '`EASTEREIG_USE_FPOLY=True` to activate fortran version.')


def _getversion():
    """ Get version from VERSION."""
    v = None
    with open(os.path.join('./eastereig', 'VERSION')) as f:
        for line in f:
            if line.startswith('version'):
                v = line.replace("'", '').split()[-1].strip()
                break
        return v


this_version = _getversion()
print('version:', this_version)

setup(
    name="eastereig",
    version=this_version,
    author="B. Nennig, M. Ghienne",
    author_email="benoit.nennig@isae-supmeca.fr, martin.ghienne@isae-supmeca.fr",
    description="A library to locate exceptional points and to reconstruct eigenvalues loci",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nennigb/EasterEig",
    include_package_data=True,
    # we can use find_packages() to automatically discover all subpackages
    packages=setuptools.find_packages(),
    # build f90 module
    ext_modules=ext_modules,
    install_requires=['numpy',
                      'scipy',
                      'sympy>=1.4',
                      'matplotlib',
                      'tqdm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
