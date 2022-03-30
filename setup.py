#!/usr/bin/env python3
from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy

ext_modules = [
    Extension("dfttools.parsers.native_openmx", ["c/generic-parser.c", "c/native_openmx.c"], include_dirs=[numpy.get_include(), "c/"]),
    Extension("dfttools.parsers.native_qe", ["c/generic-parser.c", "c/native_qe.c"], include_dirs=[numpy.get_include(), "c/"]),
]

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='dfttools',
    version='0.1.0',
    author='Artem Pulkin',
    author_email='gpulkin@gmail.com',
    packages=find_packages(),
    data_files=["requirements.txt", "test-requirements.txt"],
    url='https://github.com/pulkin/dfttools',
    license='LICENSE.txt',
    description='Tools for parsing textual data from modern DFT (quantum chemistry) packages',
    long_description=open('README.md').read(),
    ext_modules=ext_modules,
    setup_requires=[
        'numpy', 'pytest-runner',
    ],
    install_requires=requirements,
    scripts=[
        'scripts/dft-plot-bands',
        'scripts/dft-svg-structure',
        'scripts/dft-materialsproject',
        'scripts/dft-xcrysden',
    ],
)
