import pickle
import numpy
import unittest
import numericalunits

from dfttools.utypes import CrystalCell, CrystalGrid, BandsPath, BandsGrid, ReciprocalSpaceBasis, RealSpaceBasis
from dfttools.util import dumps, loads, ArrayWithUnits
from numpy import testing


def assert_standard_crystal_cell(c):
    assert isinstance(c, CrystalCell)
    assert c.vectors.units == "angstrom"


def assert_standard_real_space_grid(c):
    assert isinstance(c, CrystalGrid)
    assert c.vectors.units == "angstrom"


def assert_standard_bands_path(c):
    assert isinstance(c, BandsPath)
    assert c.vectors.units == "1/angstrom"
    assert c.values.units == "eV"
    assert c.fermi is None or c.fermi.units == "eV"


class CommonTests(unittest.TestCase):

    def test_units_remain_unchanged(self):
        b = RealSpaceBasis((1, 1), kind="orthorombic")
        testing.assert_equal(b.vectors.units, "angstrom")

        b = RealSpaceBasis(ArrayWithUnits(([1, 0], [0, 1]), units="nm"))
        testing.assert_equal(b.vectors.units, "nm")

        b = RealSpaceBasis(ArrayWithUnits([1, 1], units="nm"), kind="orthorombic")
        testing.assert_equal(b.vectors.units, "nm")
        c = RealSpaceBasis(b)
        testing.assert_equal(c.vectors.units, "nm")

        b = RealSpaceBasis(ArrayWithUnits([1, 1, 1, 0, 0, 0], units="nm"), kind="triclinic")
        testing.assert_equal(b.vectors.units, "nm")



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
            fermi=1.5 * numericalunits.eV,
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
        testing.assert_allclose(x.fermi, cell2.fermi)

    def test_save_load_json(self):
        cell = self.bs_cell

        data = dumps(cell.to_json())
        numericalunits.reset_units()
        x = BandsPath.from_json(loads(data))

        # Assert object is the same wrt numericalunits
        self.setUp()
        cell2 = self.bs_cell
        testing.assert_allclose(x.vectors, cell2.vectors)
        testing.assert_equal(x.coordinates, cell2.coordinates)
        testing.assert_allclose(x.values, cell2.values)
        testing.assert_allclose(x.fermi, cell2.fermi)

    def test_serialization_cry(self):
        serialized = self.co_cell.to_json()
        testing.assert_equal(serialized, dict(
            vectors=self.co_cell.vectors,
            meta={},
            type="dfttools.utypes.CrystalCell",
            coordinates=self.co_cell.coordinates,
            values=self.co_cell.values,
        ))

    def test_serialization_bs(self):
        serialized = self.bs_cell.to_json()
        testing.assert_equal(serialized, dict(
            vectors=self.bs_cell.vectors,
            meta={},
            type="dfttools.utypes.BandsPath",
            coordinates=self.bs_cell.coordinates,
            values=self.bs_cell.values,
            fermi=self.bs_cell.fermi,
        ))


class GridTest(unittest.TestCase):

    def setUp(self):
        x = numpy.linspace(0, 1, 2, endpoint=False)
        y = numpy.linspace(0, 1, 3, endpoint=False)
        z = numpy.linspace(0, 1, 4, endpoint=False)
        xx, yy, zz = numpy.meshgrid(x, y, z, indexing='ij')
        data = (xx ** 2 + yy ** 2 + zz ** 2) * numericalunits.eV
        self.bs_grid = BandsGrid(
            ReciprocalSpaceBasis(numpy.array((1, 2, 3)) / numericalunits.angstrom, kind='orthorombic'),
            (x, y, z),
            data,
            fermi=1.5 * numericalunits.eV,
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
        testing.assert_allclose(x.fermi, grid2.fermi)

    def test_save_load_json(self):
        grid = self.bs_grid

        data = dumps(grid.to_json())
        numericalunits.reset_units()
        x = BandsGrid.from_json(loads(data))

        # Assert object is the same wrt numericalunits
        self.setUp()
        grid2 = self.bs_grid
        testing.assert_allclose(x.vectors, grid2.vectors)
        testing.assert_equal(x.coordinates, grid2.coordinates)
        testing.assert_allclose(x.values, grid2.values)
        testing.assert_allclose(x.fermi, grid2.fermi)

    def test_save_load_json_with_conversion(self):
        grid = self.bs_grid

        data = dumps(grid.as_unitCell().to_json())
        numericalunits.reset_units()
        x = BandsPath.from_json(loads(data)).as_grid()

        # Assert object is the same wrt numericalunits
        self.setUp()
        grid2 = self.bs_grid

        testing.assert_equal(x.vectors.units, grid2.vectors.units)
        testing.assert_equal(x.values.units, grid2.values.units)

        testing.assert_allclose(x.vectors, grid2.vectors)
        testing.assert_equal(x.coordinates, grid2.coordinates)
        testing.assert_allclose(x.values, grid2.values)
        testing.assert_allclose(x.fermi, grid2.fermi)

    def test_serialization(self):
        serialized = self.bs_grid.to_json()
        testing.assert_equal(serialized, dict(
            vectors=self.bs_grid.vectors,
            meta={},
            type="dfttools.utypes.BandsGrid",
            coordinates=self.bs_grid.coordinates,
            values=self.bs_grid.values,
            fermi=self.bs_grid.fermi,
        ))

    def test_interpolate(self):
        a = self.bs_grid.interpolate_to_array(([.1, .2, .3], [.4, .5, .6]))
        assert isinstance(a, ArrayWithUnits)
        testing.assert_equal(a.units, self.bs_grid.values.units)

        a = self.bs_grid.interpolate_to_cell(([.1, .2, .3], [.4, .5, .6]))
        assert isinstance(a, BandsPath)
        assert isinstance(a.values, ArrayWithUnits)
        testing.assert_equal(a.values.units, self.bs_grid.values.units)
        assert isinstance(a.vectors, ArrayWithUnits)
        testing.assert_equal(a.vectors, self.bs_grid.vectors)
        testing.assert_equal(a.fermi, self.bs_grid.fermi)

        a = self.bs_grid.interpolate_to_path(([.1, .2, .3], [.4, .5, .6]), 3)
        assert isinstance(a, BandsPath)
        assert isinstance(a.values, ArrayWithUnits)
        testing.assert_equal(a.values.units, self.bs_grid.values.units)
        assert isinstance(a.vectors, ArrayWithUnits)
        testing.assert_equal(a.vectors, self.bs_grid.vectors)
        testing.assert_equal(a.fermi, self.bs_grid.fermi)

        a = self.bs_grid.interpolate_to_grid(([.1, .2, .3], [.4, .5, .6], [.7, .8, .9]))
        assert isinstance(a, BandsGrid)
        assert isinstance(a.values, ArrayWithUnits)
        testing.assert_equal(a.values.units, self.bs_grid.values.units)
        assert isinstance(a.vectors, ArrayWithUnits)
        testing.assert_equal(a.vectors, self.bs_grid.vectors)
        testing.assert_equal(a.fermi, self.bs_grid.fermi)
