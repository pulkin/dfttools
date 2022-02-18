import unittest

from dfttools.parsers.wannier90 import input
from ..utypes_test import assert_standard_crystal_cell, assert_standard_bands_path
import numpy
from numpy import testing

import os
import numericalunits


class Test_input0(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/wannier.input.0.testcase"), 'r') as f:
            data = f.read()
            self.parser = input(data)

    def test_cell(self):
        c = self.parser.cell()
        assert_standard_crystal_cell(c)

        testing.assert_equal(c.vectors, numpy.array((
            (-4.1380002,  0.0000000,  0.0000000,),
            (-2.0690001, -3.5836116,  0.0000000,),
            (0.0000000,  0.0000000, 28.6400012,),
        )) * numericalunits.angstrom)

        testing.assert_equal(c.coordinates, numpy.array((
            (0.3333333, 0.3333334, 0.5393333,),
            (0.3333333, 0.3333334, 0.1273333,),
            (0.3333333, 0.3333334, 0.7323333,),
            (0.3333333, 0.3333334, 0.9343333,),
            (0.6666665, 0.6666670, 0.6666667,),
            (0.6666665, 0.6666670, 0.8726667,),
            (0.6666665, 0.6666670, 0.0656667,),
            (0.6666665, 0.6666670, 0.2676667,),
            (1.0000000, 0.0000000, 0.0000000,),
            (-0.0000002, 1.0000005, 0.2060000,),
            (-0.0000002, 1.0000005, 0.3990000,),
            (0.0000000, 0.0000000, 0.7940000,),
            (0.0000000, 0.0000000, 0.6010000,),
            (0.6666665, 0.6666670, 0.4606667,),
            (0.3333333, 0.3333334, 0.3333333,),
        )))

        testing.assert_equal(c.values, ('se', 'se', 'bi', 'bi', 'se', 'se', 'bi', 'bi', 'se', 'se', 'bi', 'se',
                                        'bi', 'se', 'se'))
