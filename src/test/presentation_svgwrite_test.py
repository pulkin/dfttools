import math
import os
import tempfile
import unittest

import matplotlib
import numpy
import os.path
from numericalunits import angstrom

matplotlib.use('SVG')
from matplotlib import pyplot
from matplotlib.testing.decorators import cleanup

import svgwrite

from dfttools.types import Basis, UnitCell
from dfttools.presentation import svgwrite_unit_cell, matplotlib2svgwrite


class CellSVGTest(unittest.TestCase):

    def setUp(self):
        self.cell = UnitCell(
            Basis((2.5 * angstrom, 2.5 * angstrom, 10 * angstrom, 0, 0, .5), kind='triclinic'),
            (
                (1. / 3, 1. / 3, .5),
                (2. / 3, 2. / 3, .5),
            ),
            'C',
        ).repeated(2, 2, 2)

        self.cell2 = UnitCell(
            Basis((3.9 * angstrom / 2, 3.9 * angstrom / 2, 3.9 * angstrom / 2, .5, .5, .5), kind='triclinic'),
            (0, 0, 0),
            'Si',
        )

    def __check_dims__(self, svg, cell, window, **kwargs):
        circles = 0
        lines = 0

        for e in svg.elements:

            if isinstance(e, svgwrite.shapes.Circle):
                circles += 1

                r = e["r"]
                x = e["cx"]
                y = e["cy"]

                assert x >= r + window[0]
                assert y >= r + window[1]
                assert x <= window[2] - r
                assert y <= window[3] - r

            elif isinstance(e, svgwrite.shapes.Line):
                lines += 1

                for x, y in ((e["x1"], e["y1"]), (e["x2"], e["y2"])):
                    assert x >= window[0]
                    assert y >= window[1]
                    assert x <= window[2]
                    assert y <= window[3]

                assert e["stroke-width"] > 0

        if "show_atoms" in kwargs and kwargs["show_atoms"] == True:
            assert circles == cell.size

        if cell.size > 0 and "show_bonds" in kwargs and kwargs["show_bonds"] == True:
            assert lines > 0

    def __test_0__(self, cell, **kwargs):

        svg = svgwrite.Drawing(size=(1000, 1000))
        svgwrite_unit_cell(self.cell, svg, insert=(100, 100), size=(800, 800), **kwargs)
        self.__check_dims__(svg, self.cell, (99, 99, 901, 901))

    def test_draw_simple_xyz(self):
        for s in (True, False):
            for c in ('x', 'y', 'z'):
                for atoms in (True, False):
                    for bonds in (True, False):
                        self.__test_0__(self.cell, camera=c, show_cell=s, show_atoms=atoms, show_bonds=bonds)

    def test_draw_inplane_xyz_rotation(self):
        for c in ('x', 'y', 'z'):
            for angle in numpy.linspace(0, 2 * math.pi, 10) + 0.123:
                self.__test_0__(self.cell, camera=c, camera_top=(math.cos(angle), math.sin(angle), 0))

    def test_draw_rotation(self):
        for angle in numpy.linspace(0, 2 * math.pi, 10):
            self.__test_0__(self.cell, camera=(1, 2, 3), camera_top=(math.cos(angle), math.sin(angle), 0))

    def test_fail(self):
        with self.assertRaises(ValueError):
            self.__test_0__(self.cell, camera=(1, 2, 3), camera_top=(2, 4, 6))

    def test_save(self):
        fl = tempfile.mkstemp()[1]
        svgwrite_unit_cell(self.cell, fl)
        assert os.path.isfile(fl)
        os.remove(fl)

    def test_2(self):
        fl = tempfile.mkstemp()[1]
        svg = svgwrite_unit_cell(self.cell2, fl, show_cell=True)
        self.__check_dims__(svg, self.cell2, (-0.1, -0.1, svg["width"] + 0.1, svg["height"] + 0.1))
        assert os.path.isfile(fl)
        os.remove(fl)


class Matplotlib2SVGTest(unittest.TestCase):

    @cleanup
    def test_matplotlib(self):
        pyplot.plot([1, 2, 3], [1, 3, 2])
        svg = svgwrite.Drawing(size=(1000, 1000))
        matplotlib2svgwrite(pyplot.gcf(), svg, (100, 100), (800, 800))


class CellSVGVisualTest(unittest.TestCase):
    """
    Creates a file 'test.svg' if uncommented. Here for debugging purposes.
    """

    def setUp(self):
        self.cell = UnitCell(
            Basis((3.19 * angstrom, 3.19 * angstrom, 10 * angstrom, 0, 0, .5), kind='triclinic'),
            (
                (1. / 3, 1. / 3, .5),
                (2. / 3, 2. / 3, .6),
                (2. / 3, 2. / 3, .4),
            ),
            ('Mo', 'S', 'S'),
        ).repeated(10, 10)

    # def test_example(self):
    # svgwrite_unit_cell(self.cell, "test.svg", camera = (1,2,3), show_cell = True)
