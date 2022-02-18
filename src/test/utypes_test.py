import pickle
import numpy
import unittest
import numericalunits

from dfttools.types import CrystalCell, CrystalGrid, BandsPath, BandsGrid, ReciprocalSpaceBasis, RealSpaceBasis
from dfttools.util import dumps, loads, ArrayWithUnits, angstrom, eV, eval_nu
from numpy import testing
from pycoordinates.util import generate_path


def assert_standard_crystal_cell(c):
    assert isinstance(c, CrystalCell)
    assert c.vectors.units == "angstrom"
    for k in "total-energy", "forces":
        if k in c.meta and not isinstance(c.meta[k], ArrayWithUnits):
            raise AssertionError("Missing units in cell.meta['{}']".format(k))


def assert_standard_real_space_grid(c):
    assert isinstance(c, CrystalGrid)
    assert c.vectors.units == "angstrom"


def assert_standard_bands_path(c):
    assert isinstance(c, BandsPath)
    assert c.vectors.units == "1/angstrom"
    assert c.values.units == "eV"
    assert c.fermi is None or c.fermi.units == "eV"


class CommonTests(unittest.TestCase):

    def test_units(self):
        b = RealSpaceBasis.orthorhombic((1, 1))
        testing.assert_equal(b.vectors.units, "angstrom")

        b = RealSpaceBasis(ArrayWithUnits(([1, 0], [0, 1]), units="nm"))
        testing.assert_equal(b.vectors.units, "angstrom")

        b = RealSpaceBasis.orthorhombic(ArrayWithUnits([1, 1], units="nm"))
        testing.assert_equal(b.vectors.units, "angstrom")
        c = RealSpaceBasis(b)
        testing.assert_equal(c.vectors.units, "angstrom")

        b = RealSpaceBasis.triclinic(ArrayWithUnits([1, 1, 1], units='nm'), [0, 0, 0])
        testing.assert_equal(b.vectors.units, "angstrom")
        c = b.reciprocal
        testing.assert_equal(eval_nu(c.vectors.units), eval_nu("1/angstrom"))


class CellTest(unittest.TestCase):

    def setUp(self):
        self.a = 2.510 * numericalunits.angstrom
        self.h = self.a * (2. / 3.) ** 0.5
        self.co_cell = CrystalCell(
            ((self.a, 0, 0), (.5 * self.a, .5 * self.a * 3. ** .5, 0), (0, 0, self.h)),
            ((0., 0., 0.), (1. / 3., 1. / 3., 0.5)),
            ['Co'] * 2,
            meta={"length": angstrom(1 * numericalunits.angstrom)}
        )
        self.ia = 1. / self.a
        self.ih = 1. / self.h
        self.bs_cell = BandsPath(
            ((self.ia, 0, 0), (.5 * self.ia, .5 * self.ia * 3. ** .5, 0), (0, 0, self.ih)),
            ((0., 0., 0.), (1. / 3., 1. / 3., 0.5)),
            [3 * numericalunits.eV] * 2,
            fermi=eV(1.5 * numericalunits.eV),
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

    def test_save_load_uc(self):
        cell = self.co_cell

        data = pickle.dumps(cell)
        numericalunits.reset_units()
        x = pickle.loads(data)

        # Assert object is the same wrt numericalunits
        self.setUp()
        cell2 = self.co_cell
        testing.assert_allclose(x.vectors, cell2.vectors)
        testing.assert_equal(x.coordinates, cell2.coordinates)
        testing.assert_equal(x.values, cell2.values)
        testing.assert_allclose(x.meta["length"], cell2.meta["length"])

    def test_save_load_json(self):
        cell = self.bs_cell

        data = dumps(cell.state_dict())
        numericalunits.reset_units()
        x = BandsPath.from_state_dict(loads(data))

        # Assert object is the same wrt numericalunits
        self.setUp()
        cell2 = self.bs_cell
        testing.assert_allclose(x.vectors, cell2.vectors)
        testing.assert_equal(x.coordinates, cell2.coordinates)
        testing.assert_allclose(x.values, cell2.values)
        testing.assert_allclose(x.fermi, cell2.fermi)

    def test_serialization_cry(self):
        serialized = self.co_cell.state_dict()
        testing.assert_equal(serialized, dict(
            vectors=self.co_cell.vectors,
            meta={"length": self.co_cell.meta["length"]},
            type="dfttools.types.CrystalCell",
            coordinates=self.co_cell.coordinates,
            values=self.co_cell.values,
        ))

    def test_serialization_bs(self):
        serialized = self.bs_cell.state_dict()
        testing.assert_equal(serialized, dict(
            vectors=self.bs_cell.vectors,
            meta={},
            type="dfttools.types.BandsPath",
            coordinates=self.bs_cell.coordinates,
            values=self.bs_cell.values,
            fermi=self.bs_cell.fermi,
        ))

    def test_interpolate(self):
        c = self.bs_cell.interpolate(([.1, .2, .3], [.4, .5, .6]))
        assert isinstance(c, BandsPath)
        assert isinstance(c.values, ArrayWithUnits)
        testing.assert_equal(c.values.units, self.bs_cell.values.units)

    def test_fermi(self):
        assert self.bs_cell.fermi.units == "eV"

    def test_fermi2(self):
        b = ReciprocalSpaceBasis.orthorhombic((1./numericalunits.angstrom,) * 3)
        coords = b.transform_from_cartesian(generate_path(b.transform_to_cartesian((
            (0, 0, 0),
            (0, 0, .5),
            (.5, .5, .5),
        )), 100))
        bands = (numpy.linalg.norm(coords, axis=-1) ** 2 + 1)[:, numpy.newaxis] * [[-2, 1]]
        bands = BandsPath(b, coords, bands, fermi=-1)
        self.assertEqual(bands.nocc, 1)
        self.assertEqual(bands.nvirt, 1)
        self.assertEqual(bands.gapped, True)
        self.assertEqual(bands.vbt, -2)
        self.assertEqual(bands.cbb, 1)
        self.assertEqual(bands.gap, 3)

        b = bands.stick_fermi("vbt")
        testing.assert_allclose(b.vbt, b.fermi)

        b = bands.stick_fermi("cbb")
        testing.assert_allclose(b.cbb, b.fermi)

        b = bands.stick_fermi("midgap")
        testing.assert_allclose(.5 * (b.vbt + b.cbb), b.fermi)

        b = bands.canonize_fermi()
        testing.assert_allclose(b.fermi, 0)
        testing.assert_allclose(b.vbt, -1.5)
        testing.assert_allclose(b.cbb, 1.5)


class GridTest(unittest.TestCase):

    def setUp(self):
        x = numpy.linspace(0, 1, 2, endpoint=False)
        y = numpy.linspace(0, 1, 3, endpoint=False)
        z = numpy.linspace(0, 1, 4, endpoint=False)
        xx, yy, zz = numpy.meshgrid(x, y, z, indexing='ij')
        data = (xx ** 2 + yy ** 2 + zz ** 2) * numericalunits.eV
        self.bs_grid = BandsGrid(
            ReciprocalSpaceBasis.orthorhombic(numpy.array((1, 2, 3)) / numericalunits.angstrom),
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

        data = dumps(grid.state_dict())
        numericalunits.reset_units()
        x = BandsGrid.from_state_dict(loads(data))

        # Assert object is the same wrt numericalunits
        self.setUp()
        grid2 = self.bs_grid
        testing.assert_allclose(x.vectors, grid2.vectors)
        testing.assert_equal(x.coordinates, grid2.coordinates)
        testing.assert_allclose(x.values, grid2.values)
        testing.assert_allclose(x.fermi, grid2.fermi)

    def test_save_load_json_with_conversion(self):
        grid = self.bs_grid

        data = dumps(grid.as_cell().state_dict())
        numericalunits.reset_units()
        x = BandsPath.from_state_dict(loads(data)).as_grid()

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
        serialized = self.bs_grid.state_dict()
        testing.assert_equal(serialized, dict(
            vectors=self.bs_grid.vectors,
            meta={},
            type="dfttools.types.BandsGrid",
            coordinates=self.bs_grid.coordinates,
            values=self.bs_grid.values,
            fermi=self.bs_grid.fermi,
        ))

    def test_interpolate(self):
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
