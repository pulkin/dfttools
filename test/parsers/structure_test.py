import os
import unittest

import numericalunits
import numpy
from dfttools.parsers.structure import xsf, cube, cif
from ..utypes_test import assert_standard_crystal_cell, assert_standard_real_space_grid
from numpy import testing


class Test_xsf0(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/structure.xsf.0.testcase"),
                  'r') as f:
            self.parser = xsf(f.read())

    def test_valid_header(self):
        assert xsf.valid_header(self.parser.parser.string[:1000])

    def test_cells(self):
        c = self.parser.cells()

        assert len(c) == 1
        c = c[0]

        assert_standard_crystal_cell(c)
        testing.assert_allclose(c.vectors, numpy.array((
            (2.71, 2.71, 0.),
            (2.71, 0., 2.71),
            (0., 2.71, 2.71),
        )) * numericalunits.angstrom)

        testing.assert_allclose(c.cartesian, numpy.array((
            (0, 0, 0),
            (1.355, -1.355, -1.355)
        )) * numericalunits.angstrom)

        testing.assert_array_equal(c.values[:2], ("16", "30"))


class Test_xsf1(unittest.TestCase):

    def test_valid_header(self):
        assert xsf.valid_header(self.parser.parser.string[:1000])

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/structure.xsf.1.testcase"),
                  'r') as f:
            self.parser = xsf(f.read())

    def test_cells(self):
        c = self.parser.cells()

        assert len(c) == 2

        for cc in c:
            assert_standard_crystal_cell(cc)
            testing.assert_allclose(cc.vectors, numpy.array((
                (0., 2.71, 2.71),
                (2.71, 0., 2.71),
                (2.71, 2.71, 0.),
            )) * numericalunits.angstrom)

            testing.assert_array_equal(cc.values[:2], ("16", "30"))

        testing.assert_allclose(c[0].cartesian, numpy.array((
            (0, 0, 0),
            (1.355, -1.355, -1.355)
        )) * numericalunits.angstrom)

        testing.assert_allclose(c[1].cartesian, numpy.array((
            (0, 0, 0),
            (1.255, -1.255, -1.255)
        )) * numericalunits.angstrom)


class Test_xsf2(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/structure.xsf.2.testcase"),
                  'r') as f:
            self.parser = xsf(f.read())

    def test_valid_header(self):
        assert xsf.valid_header(self.parser.parser.string[:1000])

    def test_cells(self):
        c = self.parser.cells()

        assert len(c) == 2

        for cc in c:
            assert_standard_crystal_cell(cc)
            testing.assert_array_equal(cc.values[:2], ("16", "30"))

        testing.assert_allclose(c[0].vectors, numpy.array((
            (2.71, 2.71, 0.),
            (2.71, 0., 2.71),
            (0., 2.71, 2.71),
        )) * numericalunits.angstrom)

        testing.assert_allclose(c[1].vectors, numpy.array((
            (2.981, 2.981, 0.),
            (2.981, 0., 2.981),
            (0., 2.981, 2.981),
        )) * numericalunits.angstrom)

        testing.assert_allclose(c[0].cartesian, numpy.array((
            (0, 0, 0),
            (1.355, -1.355, -1.355)
        )) * numericalunits.angstrom)

        testing.assert_allclose(c[1].cartesian, numpy.array((
            (0, 0, 0),
            (1.5905, -1.5905, -1.5905)
        )) * numericalunits.angstrom)


class Test_xsf3(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/structure.xsf.3.testcase"),
                  'r') as f:
            self.parser = xsf(f.read())

    def test_grids(self):
        c = self.parser.grids()

        assert len(c) == 3

        assert_standard_real_space_grid(c[0])
        testing.assert_equal(c[0].vectors, numpy.array((
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )) * numericalunits.angstrom)

        testing.assert_equal(c[0].values, numpy.array((
            (0.000, 1.000, 2.000, 5.196),
            (1.000, 1.414, 2.236, 5.292),
            (2.000, 2.236, 2.828, 5.568),
            (3.000, 3.162, 3.606, 6.000),
        )).transpose())

        assert len(c[0].coordinates) == 2

        for i in range(2):
            testing.assert_equal(c[0].coordinates[i], numpy.linspace(0, 1, 4, endpoint=False))

        testing.assert_equal(c[0].meta["xsf-grid-origin"], [0, 0, 0])
        assert c[0].meta["xsf-block-name"] == "my_first_example_of_2D_datagrid"
        assert c[0].meta["xsf-grid-name"] == "this_is_2Dgrid#1"

        testing.assert_equal(c[2].vectors, numpy.eye(3) * numericalunits.angstrom)

        testing.assert_equal(c[2].values, numpy.array((
            (
                (0.000, 1.000, 2.000, 5.196),
                (1.000, 1.414, 2.236, 5.292),
                (2.000, 2.236, 2.828, 5.568),
                (3.000, 3.162, 3.606, 6.000),
            ), (
                (1.000, 1.414, 2.236, 5.292),
                (1.414, 1.732, 2.449, 5.385),
                (2.236, 2.449, 3.000, 5.657),
                (3.162, 3.317, 3.742, 6.083),
            ), (
                (2.000, 2.236, 2.828, 5.568),
                (2.236, 2.449, 3.000, 5.657),
                (2.828, 3.000, 3.464, 5.916),
                (3.606, 3.742, 4.123, 6.325),
            ), (
                (3.000, 3.162, 3.606, 6.000),
                (3.162, 3.317, 3.742, 6.083),
                (3.606, 3.742, 4.123, 6.325),
                (4.243, 4.359, 4.690, 6.708),
            ),
        )).swapaxes(0, 2))

        assert len(c[2].coordinates) == 3

        for i in range(3):
            testing.assert_equal(c[2].coordinates[i], numpy.linspace(0, 1, 4, endpoint=False))


class Test_xsf4(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/structure.xsf.4.testcase"),
                  'r') as f:
            self.parser = xsf(f.read())

    def test_grid(self):
        c = self.parser.grids()
        assert len(c) == 1

        assert_standard_real_space_grid(c[0])
        testing.assert_equal(c[0].vectors, numpy.array((
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )) * numericalunits.angstrom)

        testing.assert_equal(c[0].values, numpy.array((
            (0.000, 1.000, 2.000, 5.196),
            (1.000, 1.414, 2.236, 5.292),
            (2.000, 2.236, 2.828, 5.568),
            (3.000, 3.162, 3.606, 6.000),
        )).transpose())

        assert len(c[0].coordinates) == 2

        for i in range(2):
            testing.assert_equal(c[0].coordinates[i], numpy.linspace(0, 1, 4, endpoint=False))

        testing.assert_equal(c[0].meta["xsf-grid-origin"], [0, 0, 0])
        assert c[0].meta["xsf-block-name"] == "my_first_example_of_2D_datagrid"
        assert c[0].meta["xsf-grid-name"] == "this_is_2Dgrid#1"


class Test_cube0(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/structure.cube.0.testcase"),
                  'r') as f:
            self.parser = cube(f.read())

    def test_grid(self):
        c = self.parser.grid()

        assert_standard_real_space_grid(c)
        testing.assert_allclose(c.vectors, numpy.diag((4., 5., 6.)) * 0.283459 * numericalunits.aBohr, rtol=1e-12)

        test = numpy.array((((1., 2., 3., 4., 5., 6.),) * 5,) * 4)
        test[1, 2, 3] = 9
        test[-1, -1, -1] = 9
        testing.assert_equal(c.values, test)

        assert len(c.coordinates) == 3

        testing.assert_equal(c.coordinates[0], numpy.linspace(0, 1, 4, endpoint=False))
        testing.assert_equal(c.coordinates[1], numpy.linspace(0, 1, 5, endpoint=False))
        testing.assert_equal(c.coordinates[2], numpy.linspace(0, 1, 6, endpoint=False))

    def test_cell(self):
        c = self.parser.cell()

        assert_standard_crystal_cell(c)
        testing.assert_allclose(c.cartesian, numpy.array((
            (5.570575, 5.669178, 5.593517),
            (5.562867, 5.669178, 7.428055),
            (7.340606, 5.669178, 5.111259)
        )) * numericalunits.aBohr)

        assert c.values[0].lower() == 'o'
        assert c.values[1].lower() == 'h'
        assert c.values[2].lower() == 'h'


class Test_cube1(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/structure.cube.1.testcase"),
                  'r') as f:
            self.parser = cube(f.read())

    def test_grid(self):
        c = self.parser.grid()
        assert_standard_real_space_grid(c)

        testing.assert_allclose(c.vectors, numpy.diag((4., 5., 6.)) * 0.283459 * numericalunits.angstrom, rtol=1e-12)

    def test_cell(self):
        c = self.parser.cell()

        assert_standard_crystal_cell(c)
        testing.assert_allclose(c.cartesian, numpy.array((
            (5.570575, 5.669178, 5.593517),
            (5.562867, 5.669178, 7.428055),
            (7.340606, 5.669178, 5.111259)
        )) * numericalunits.angstrom)

        assert c.values[0].lower() == 'o'
        assert c.values[1].lower() == 'h'
        assert c.values[2].lower() == 'h'


class Test_cif0(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/structure.cif.0.testcase"),
                  'r') as f:
            self.parser = cif(f.read())

    def test_cell(self):
        cells = self.parser.cells()
        assert len(cells) == 1
        cell = cells[0]

        assert_standard_crystal_cell(cell)

        m = cell.vectors.dot(cell.vectors.T) / numericalunits.angstrom ** 2
        a, b, c = 9.52700, 8.65870, 5.13060
        g = numpy.cos(105.06700 * numpy.pi / 180)
        testing.assert_allclose(m, (
            (a*a, 0, a*c*g),
            (0, b*b, 0),
            (a*c*g, 0, c*c),
        ), atol=2e-6)

        testing.assert_allclose(cell.coordinates, (
            (0.00000, 0.91060, 0.25000),
            (0.00000, 0.30588, 0.25000),
            (0.28614, 0.09547, 0.22733),
            (0.11520, 0.08809, 0.14222),
            (0.35914, 0.25645, 0.32137),
            (0.35342, 0.02336, 0.98756),
        ))

        testing.assert_equal(cell.values, ("MgM1", "CaM2", "Si", "O1", "O2", "O3"))


class Test_cif1(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/structure.cif.1.testcase"),
                  'r') as f:
            self.parser = cif(f.read())

    def test_cell(self):
        cells = self.parser.cells()
        assert len(cells) == 1
        cell = cells[0]

        assert_standard_crystal_cell(cell)

        m = cell.vectors.dot(cell.vectors.T) / numericalunits.angstrom ** 2
        a, b, c = 12.564575, 12.24754, 30.445
        g = numpy.cos(120.2902 * numpy.pi / 180)
        testing.assert_allclose(m, (
            (a*a, a*b*g, 0),
            (a*b*g, b*b, 0),
            (0, 0, c*c),
        ), atol=1e-5)

        testing.assert_allclose(cell.coordinates, (
            (0.00385969,  0.00269148,  0.51418327),
            (0.25165061,  0.12384214,  0.32650931),
        ))

        testing.assert_equal(cell.values, ("bi", "se"))
