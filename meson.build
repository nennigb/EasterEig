project(
    'eastereig',
    version: run_command('eastereig/version.py', check: true).stdout().strip(),
    license: 'GNU General Public License v3 (GPLv3)',
    meson_version: '>=1.1.0',
    default_options: ['warning_level=2', 'buildtype=release'],
)

# https://mesonbuild.com/Build-options.html
# works with `meson.options` file
use_fpoly = get_option('use_fpoly')

# Load meson python module
py_mod = import('python')
py = py_mod.find_installation(pure: false)

if use_fpoly == true
    add_languages(['c', 'fortran'])
    # https://github.com/numpy/numpy/issues/25000
    quadmath_dep = meson.get_compiler('fortran').find_library(
        'quadmath',
        required: false,
    )

    # Get python dependency, need to be after langage definition
    py_dep = py.dependency()

    # Get include from numpy and f2py
    # based on https://numpy.org/doc/stable/f2py/buildtools/meson.html
    incdir_numpy = run_command(
        py,
        [
            '-c',
            'import os; os.chdir(".."); import numpy; print(numpy.get_include())',
        ],
        check: true,
    ).stdout().strip()

    incdir_f2py = run_command(
        py,
        [
            '-c',
            'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())',
        ],
        check: true,
    ).stdout().strip()

    inc_np = include_directories(incdir_numpy, incdir_f2py)

    # From https://numpy.org/doc/stable/f2py/buildtools/index.html
    # For Fortran 90 programs, f2py generates for an input file blah.f90
    #  blahmodule.c
    #  blah-f2pywrappers.f, makes subroutines for functions, it rewrites assumed shape arrays as automatic arrays (empty for pypolsys).
    #  blah-f2pywrappers2.f90, handles code which is subdivided into modules
    _fpoly_module = custom_target(
        'fpoly',
        output: [
            'fpolymodule.c',
            #        'fpoly-f2pywrappers2.f90',
            'fpoly-f2pywrappers.f',
        ],
        input: ['eastereig/fpoly/fpolyval.f90'],
        command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'fpoly', '--lower'],
    )
    # fpoly extension own src files
    src_files = [_fpoly_module, 'eastereig/fpoly/fpolyval.f90']

    deps = [py_dep, quadmath_dep]

    ## Assumes that f2py was run before in _polsys_module
    py.extension_module(
        'eastereig.fpoly._fpolyval',
        src_files,
        incdir_f2py / 'fortranobject.c',
        include_directories: inc_np,
        # No problem if lapack_dep is not found, meson just ignore it
        dependencies: deps,
        install: true,
        native: true,
        subdir: 'eastereig/fpoly',  # Folder relative to site-packages to install to
    )
endif


# Add pure python files
python_sources = [
    'eastereig/__init__.py',
    'eastereig/__main__.py',
    'eastereig/_petsc_utils.py',
    'eastereig/adapter.py',
    'eastereig/charpol.py',
    'eastereig/eig.py',
    'eastereig/eigSolvers.py',
    'eastereig/ep.py',
    'eastereig/lda_func.py',
    'eastereig/loci.py',
    'eastereig/op.py',
    'eastereig/options.py',
    'eastereig/tests/test_charpol.py',
    'eastereig/tests/test_multiindex.py',
    'eastereig/utils.py',
    'eastereig/version.py',
    'eastereig/examples/__init__.py',
    'eastereig/examples/ThreeDoF.py',
    'eastereig/examples/WGadmitance_numpy_mv.py',
    'eastereig/examples/WGadmitance_petsc_mv.py',
    'eastereig/examples/WGadmitance_scipy_mv.py',
    'eastereig/examples/WGimpedance_numpy.py',
    'eastereig/examples/WGimpedance_petsc.py',
    'eastereig/examples/WGimpedance_scipysp.py',
    'eastereig/examples/toy_3dof_2params.py',
    'eastereig/fpoly/__init__.py',
    'eastereig/fpoly/_fpoly.py',
    'eastereig/fpoly/_fpolyval_full_py.py',
]

# Install pure Python
# see https://mesonbuild.com/Python-module.html
py.install_sources(
    python_sources,
    pure: false,  # Will be installed next to binaries
    #subdir: 'eastereig',  # Folder relative to site-packages to install to
    preserve_path: true,
)

## Add package_data used for tests as source files
#install_dir = py.get_install_dir(subdir: 'pypolsys/examples/data')
#install_data('pypolsys/examples/data/toy_model.npz', install_dir: install_dir)  # / 'examples' / 'data'

