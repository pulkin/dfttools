import unittest
import os

from dfttools.parsers.tools import jsons
from ..utypes_test import assert_standard_crystal_cell
from numpy import testing
import numericalunits


class Test_Structure0(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/dfttools.0.testcase"), 'r') as f:
            self.parser = jsons(f.read())

    def test_structure(self):
        cell = self.parser.unitCells()
        assert_standard_crystal_cell(cell)

        a = 4.138
        c = 28.64
        mu = 0.399
        nu = 0.206

        testing.assert_allclose(cell.vectors / numericalunits.angstrom, (
            (-a / 2, -3. ** .5 * a / 6, c / 3),
            (a / 2, -3. ** .5 * a / 6, c / 3),
            (0, -3. ** .5 * a / 3, c / 3),
        ))
        testing.assert_allclose(cell.coordinates, (
            (0,) * 3,
            (nu,) * 3,
            (-nu,) * 3,
            (mu,) * 3,
            (-mu,) * 3,
        ))
        testing.assert_equal(cell.values, ("se",) * 3 + ("bi",) * 2)

