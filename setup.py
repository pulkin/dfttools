#!/usr/bin/env python3
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy


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
    long_description_content_type="text/markdown",
    ext_modules=cythonize([
        Extension("dfttools.parsers.native_qe", ["cython/native_qe.pyx"], include_dirs=[numpy.get_include()]),
        Extension("dfttools.parsers.native_openmx", ["cython/native_openmx.pyx"], include_dirs=[numpy.get_include()]),
        "cython/fastparse.pyx",
    ]),
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
