import os
import unittest

import numericalunits
import numpy
from dfttools.parsers.materialsproject import jsonr
from numpy import testing


class Test_structure0(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "cases/materialsproject.unitcells.0.testcase"), 'r') as f:
            self.parser = jsonr(f.read())

    def test_unitCells(self):
        cells = self.parser.unitCells()

        assert len(cells) == 34

        c = cells[0]
        assert c.units_aware()
        testing.assert_equal(c.vectors, numpy.array([
            [1.4119197, 2.0624061, 7.43608817],
            [-1.0453002, 1.48076613, 2.58860205],
            [-1.22470273, -2.07629031, -3.32969265],
        ]) * numericalunits.angstrom)
        testing.assert_equal(c.coordinates, [
            [0.83281091, 0.33450064, 0.99999999],
            [0.16718909, 0.66549936, 1e-08],
        ])
        testing.assert_equal(c.values, ["C"] * 2)

        c = cells[-1]
        assert c.units_aware()
        testing.assert_equal(c.vectors, numpy.array([
            [2.46801892, 6.328e-05, 0.0],
            [-1.23410443, 2.13733939, 0.0],
            [0.0, 0.0, 19.99829344],
        ]) * numericalunits.angstrom)
        testing.assert_equal(c.coordinates, [
            [0.83333794, 0.66666694, 0.0],
            [0.16666206, 0.33333306, 0.0],
        ])
        testing.assert_equal(c.values, ["C"] * 2)
