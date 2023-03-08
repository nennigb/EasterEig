# -*- coding: utf-8 -*-
import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()


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

setuptools.setup(
    name="eastereig",
    version=this_version,
    author="B. Nennig, M. Ghienne",
    author_email="benoit.nennig@supmeca.fr, martin.ghienne@supmeca.fr",
    description="A library to locate exceptional points and to reconstruct eigenvalues loci",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nennigb/EasterEig",
    include_package_data=True,
    # we can use find_packages() to automatically discover all subpackages
    packages=setuptools.find_packages(),
    install_requires=['numpy',
                      'scipy',
                      'matplotlib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
