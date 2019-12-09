# -*- coding: utf-8 -*-
import setuptools
import subprocess


with open("README.md", "r") as fh:
    long_description = fh.read()

# get version from git
#version=subprocess.run(['git', 'describe', '--tags'],stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n','').replace('v','')

setuptools.setup(
    name="eastereig",
    version="1.0",
    author="B. Nennig, M. Ghienne",
    author_email="benoit.nennig@supmeca.fr, martin.ghienne@supmeca.fr",
    description="A library to locate exceptional points and to reconstruct eigenvalues loci",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nennigb/EasterEig",
    packages=['eastereig'],#setuptools.find_packages(), # we can use find_packages() to automatically discover all packages and subpackages
    install_requires=['numpy',
                      'scipy',
                      'matplotlib<=3.0.0'], # max version for python3.5
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: GPL 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5', # tested with python 3.5 may works with previous py3 version...
)
