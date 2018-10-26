import pickle
import numpy
import unittest
import numericalunits
import json

from dfttools.utypes import Basis, Grid, CrystalCell, BandsPath, BandsGrid, eval_nu
from numpy import testing


def assert_standard_crystal_cell(c):
    assert isinstance(c, CrystalCell)
    assert c.units["vectors"] == "angstrom"


def assert_standard_bands_path(c):
    assert isinstance(c, BandsPath)
    assert c.units["vectors"] == "1/angstrom"
    assert c.units["values"] == "eV"


def assert_standard_real_space_grid(c):
    assert isinstance(c, Grid)
    assert c.units["vectors"] == "angstrom"


class EvalNUTest(unittest.TestCase):

    def test_nu(self):
        with self.assertRaises(ValueError):
            eval_nu("")
        with self.assertRaises(ValueError):
            eval_nu("nonexistent_unit")
        testing.assert_equal(eval_nu("angstrom"), numericalunits.angstrom)
        eva = numericalunits.eV / numericalunits.angstrom
        for i in ("eV/angstrom", "eV /angstrom", "eV/ angstrom", "eV / angstrom", "eV/angstrom ", " eV/angstrom",
                  " eV  /   angstrom  "):
            testing.assert_equal(eval_nu(i), eva)

        with self.assertRaises(ValueError):
            eval_nu("eV/nonexistent_unit")
        testing.assert_equal(eval_nu("1/angstrom"), 1./numericalunits.angstrom)


class BasisTest(unittest.TestCase):

    def test_save_load(self):
        x = old = Basis(
            (numericalunits.angstrom,) * 3,
            kind='orthorombic',
            units=dict(vectors='angstrom'),
        )
        data = pickle.dumps(x)
        numericalunits.reset_units()
        x = pickle.loads(data)
        # Assert object changed
        assert x != old
        testing.assert_allclose(x.vectors, numpy.eye(3) * numericalunits.angstrom)

    def test_save_load_json(self):
        x = old = Basis(
            (numericalunits.angstrom,) * 3,
            kind='orthorombic',
            units=dict(vectors='angstrom'),
        )
        data = json.dumps(x.to_json())
        numericalunits.reset_units()
        x = Basis.from_json(json.loads(data))
        # Assert object changed
        assert x != old
        testing.assert_allclose(x.vectors, numpy.eye(3) * numericalunits.angstrom)


class CellTest(unittest.TestCase):

    def setUp(self):
        self.a = 2.510 * numericalunits.angstrom
        self.h = self.a * (2. / 3.) ** 0.5
        self.co_cell = CrystalCell(
            ((self.a, 0, 0), (.5 * self.a, .5 * self.a * 3. ** .5, 0), (0, 0, self.h)),
            ((0., 0., 0.), (1. / 3., 1. / 3., 0.5)),
            'Co',
        )
        self.ia = 1. / self.a
        self.ih = 1. / self.h
        self.bs_cell = BandsPath(
            ((self.ia, 0, 0), (.5 * self.ia, .5 * self.ia * 3. ** .5, 0), (0, 0, self.ih)),
            ((0., 0., 0.), (1. / 3., 1. / 3., 0.5)),
            [3 * numericalunits.eV],
        )

    def test_save_load(self):
        cell = self.bs_cell

        data = pickle.dumps(cell)
        numericalunits.reset_units()
        x = pickle.loads(data)

        # Assert object is the same wrt numericalunits
        self.setUp()
        cell2 = self.bs_cell
        testing.assert_allclose(x.vectors, cell2.vectors)
        testing.assert_equal(x.coordinates, cell2.coordinates)
        testing.assert_allclose(x.values, cell2.values)

    def test_save_load_json(self):
        cell = self.bs_cell

        data = json.dumps(cell.to_json())
        numericalunits.reset_units()
        x = BandsPath.from_json(json.loads(data))

        # Assert object is the same wrt numericalunits
        self.setUp()
        cell2 = self.bs_cell
        testing.assert_allclose(x.vectors, cell2.vectors)
        testing.assert_equal(x.coordinates, cell2.coordinates)
        testing.assert_allclose(x.values, cell2.values)


class GridTest(unittest.TestCase):

    def setUp(self):
        x = numpy.linspace(0, 1, 2, endpoint=False)
        y = numpy.linspace(0, 1, 3, endpoint=False)
        z = numpy.linspace(0, 1, 4, endpoint=False)
        xx, yy, zz = numpy.meshgrid(x, y, z, indexing='ij')
        data = (xx ** 2 + yy ** 2 + zz ** 2) * numericalunits.eV
        self.bs_grid = BandsGrid(
            Basis(numpy.array((1, 2, 3)) / numericalunits.angstrom, kind='orthorombic'),
            (x, y, z),
            data,
        )

    def test_pickle_units(self):
        grid = self.bs_grid

        data = pickle.dumps(grid)
        numericalunits.reset_units()
        x = pickle.loads(data)

        # Assert object is the same wrt numericalunits
        self.setUp()
        grid2 = self.bs_grid
        testing.assert_allclose(x.vectors, grid2.vectors)
        testing.assert_equal(x.coordinates, grid2.coordinates)
        testing.assert_allclose(x.values, grid2.values)

    def test_save_load_json(self):
        grid = self.bs_grid

        data = json.dumps(grid.to_json())
        numericalunits.reset_units()
        x = BandsGrid.from_json(json.loads(data))

        # Assert object is the same wrt numericalunits
        self.setUp()
        grid2 = self.bs_grid
        testing.assert_allclose(x.vectors, grid2.vectors)
        testing.assert_equal(x.coordinates, grid2.coordinates)
        testing.assert_allclose(x.values, grid2.values)

    def test_save_load_json_with_conversion(self):
        grid = self.bs_grid

        data = json.dumps(grid.as_unitCell().to_json())
        numericalunits.reset_units()
        x = BandsPath.from_json(json.loads(data)).as_grid()

        # Assert object is the same wrt numericalunits
        self.setUp()
        grid2 = self.bs_grid
        testing.assert_allclose(x.vectors, grid2.vectors)
        testing.assert_equal(x.coordinates, grid2.coordinates)
        testing.assert_allclose(x.values, grid2.values)
