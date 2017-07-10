from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

cmdclass = {"build_ext":build_ext}
ext_modules = [ ]

if use_cython:
    ext_modules += [
        Extension("dfttools.blochl", [ "cython/blochl.pyx" ]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("dfttools.blochl", [ "cython/blochl.c" ]),
    ]
    
ext_modules += [
    Extension("dfttools.parsers.native_openmx", [ "c/generic-parser.c", "c/native_openmx.c" ]),
    Extension("dfttools.parsers.native_qe", [ "c/generic-parser.c", "c/native_qe.c" ]),
]

setup(
    name='DFT Parsing Tools',
    version='0.0.0',
    author='Artem Pulkin',
    author_email='artem.pulkin@epfl.ch',
    packages=['dfttools', 'dfttools.parsers'],
    test_suite="nose.collector",
    tests_require="nose",
    url='http://pypi.python.org/pypi/DFTTools/',
    license='LICENSE.txt',
    description='Tools for parsing textual data from modern DFT codes',
    long_description=open('README.md').read(),
    cmdclass = cmdclass,
    ext_modules=ext_modules,
    setup_requires=[
        'numpy',
    ],
    install_requires=[
        'scipy',
        'numericalunits',
        'matplotlib',
        'svgwrite',
    ],
    scripts=[
        'scripts/dft-plot-bands',
        'scripts/dft-svg-structure',
    ],
)

