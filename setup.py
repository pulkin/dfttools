#!/usr/bin/env python3
from setuptools import setup
from setuptools.extension import Extension
import numpy

ext_modules = [
    Extension("dfttools.parsers.native_openmx", ["c/generic-parser.c", "c/native_openmx.c"], include_dirs=[numpy.get_include(), "c/"]),
    Extension("dfttools.parsers.native_qe", ["c/generic-parser.c", "c/native_qe.c"], include_dirs=[numpy.get_include(), "c/"]),
]

setup(
    name='dfttools',
    version='0.1.0',
    author='Artem Pulkin',
    author_email='gpulkin@gmail.com',
    packages=['dfttools', 'dfttools.parsers'],
    test_suite="nose.collector",
    tests_require="nose",
    url='http://pypi.python.org/pypi/DFTTools/',
    license='LICENSE.txt',
    description='Tools for parsing textual data from modern DFT codes',
    long_description=open('README.md').read(),
    ext_modules=ext_modules,
    setup_requires=[
        'numpy',
    ],
    install_requires=[
        'scipy',
        'numericalunits',
        'matplotlib',
        'svgwrite',
        'requests',
    ],
    scripts=[
        'scripts/dft-plot-bands',
        'scripts/dft-svg-structure',
        'scripts/dft-materialsproject',
        'scripts/dft-xcrysden',
    ],
)
