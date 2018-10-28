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
    assert c.units["fermi"] == "eV"


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

    def setUp(self):
        self.basis = Basis(
            numpy.array((1, 2, 3)) * numericalunits.angstrom,
            kind='orthorombic',
            meta={"key": "value", "another": 3. / numericalunits.eV},
            units=dict(vectors="angstrom", meta_another="1/eV"),
        )

    def test_save_load(self):
        basis = self.basis

        data = pickle.dumps(basis)
        numericalunits.reset_units()
        x = pickle.loads(data)

        # Assert object is the same wrt numericalunits
        self.setUp()
        basis2 = self.basis
        testing.assert_allclose(x.vectors, basis2.vectors)
        testing.assert_allclose(x.meta["another"], basis2.meta["another"])

    def test_save_load_json(self):
        basis = self.basis

        data = json.dumps(basis.to_json())
        numericalunits.reset_units()
        x = Basis.from_json(json.loads(data))

        # Assert object is the same wrt numericalunits
        self.setUp()
        basis2 = self.basis
        testing.assert_allclose(x.vectors, basis2.vectors)
        testing.assert_allclose(x.meta["another"], basis2.meta["another"])

    def test_serialization(self):
        serialized = self.basis.to_json()
        testing.assert_equal(serialized, dict(
            vectors=(self.basis.vectors / numericalunits.angstrom).tolist(),
            meta=dict(key="value", another=self.basis.meta["another"] / (1. / numericalunits.eV)),
            type="dfttools.utypes.Basis",
            units=dict(vectors="angstrom", meta_another="1/eV"),
        ))


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

        data = json.dumps(cell.to_json())
        numericalunits.reset_units()
        x = BandsPath.from_json(json.loads(data))

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
            vectors=(self.co_cell.vectors / numericalunits.angstrom).tolist(),
            meta={},
            type="dfttools.utypes.CrystalCell",
            units=dict(vectors="angstrom"),
            coordinates=self.co_cell.coordinates.tolist(),
            values=self.co_cell.values.tolist(),
        ))

    def test_serialization_bs(self):
        serialized = self.bs_cell.to_json()
        testing.assert_equal(serialized, dict(
            vectors=(self.bs_cell.vectors / (1. / numericalunits.angstrom)).tolist(),
            meta={},
            type="dfttools.utypes.BandsPath",
            units=dict(vectors="1/angstrom", values="eV", fermi="eV"),
            coordinates=self.bs_cell.coordinates.tolist(),
            values=(self.bs_cell.values / numericalunits.eV).tolist(),
            fermi=self.bs_cell.fermi / numericalunits.eV,
        ))


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

        data = json.dumps(grid.to_json())
        numericalunits.reset_units()
        x = BandsGrid.from_json(json.loads(data))

        # Assert object is the same wrt numericalunits
        self.setUp()
        grid2 = self.bs_grid
        testing.assert_allclose(x.vectors, grid2.vectors)
        testing.assert_equal(x.coordinates, grid2.coordinates)
        testing.assert_allclose(x.values, grid2.values)
        testing.assert_allclose(x.fermi, grid2.fermi)

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
        testing.assert_allclose(x.fermi, grid2.fermi)

    def test_serialization(self):
        serialized = self.bs_grid.to_json()
        testing.assert_equal(serialized, dict(
            vectors=(self.bs_grid.vectors / (1. / numericalunits.angstrom)).tolist(),
            meta={},
            type="dfttools.utypes.BandsGrid",
            units=dict(vectors="1/angstrom", values="eV", fermi="eV"),
            coordinates=tuple(i.tolist() for i in self.bs_grid.coordinates),
            values=(self.bs_grid.values / numericalunits.eV).tolist(),
            fermi=self.bs_grid.fermi / numericalunits.eV,
        ))
