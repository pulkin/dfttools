import math
import os
import tempfile
import unittest
from collections import defaultdict

import matplotlib
import numpy
import os.path
from numericalunits import angstrom

matplotlib.use('SVG')
from matplotlib import pyplot
from matplotlib.testing.decorators import cleanup

import svgwrite

from dfttools.types import Basis, Cell
from dfttools.data import element_color_convention
from dfttools.presentation import svgwrite_unit_cell, matplotlib2svgwrite


class CellSVGTest(unittest.TestCase):

    def setUp(self):
        self.cell = Cell(
            Basis.triclinic((2.5 * angstrom, 2.5 * angstrom, 10 * angstrom), (0, 0, .5)),
            (
                (1. / 3, 1. / 3, .5),
                (2. / 3, 2. / 3, .5),
            ),
            ['C'] * 2,
        ).repeated(2, 2, 2)

        self.cell2 = Cell(
            Basis.triclinic((3.9 * angstrom / 2, 3.9 * angstrom / 2, 3.9 * angstrom / 2), (.5, .5, .5)),
            (0, 0, 0),
            ['Si'],
        )

    def __check_svg__(self, svg, cell, **kwargs):
        lines = 0

        atoms = defaultdict(int)
        for i in cell.values:
            atoms["rgb({:d},{:d},{:d})".format(*element_color_convention[i.lower()])] += 1

        def recursive_traversal(e):
            yield e
            if isinstance(e, svgwrite.base.BaseElement):
                for i in e.elements:
                    yield from recursive_traversal(i)

        for e in recursive_traversal(svg):
            if isinstance(e, svgwrite.shapes.Circle):
                atoms[e["fill"]] -= 1

            elif isinstance(e, svgwrite.shapes.Line):
                lines += 1
                self.assertGreater(e["stroke-width"], 0)

        # Default is True
        if "show_atoms" not in kwargs or kwargs["show_atoms"] is True:
            deviation = {k: v for k, v in atoms.items() if v != 0}
            self.assertEqual(deviation, {})

        # Default is True
        if cell.size > 0 and ("show_bonds" not in kwargs or kwargs["show_bonds"] is True):
            assert lines > 0

    def __test_0__(self, cell, **kwargs):
        svg = svgwrite.Drawing()
        svgwrite_unit_cell(cell, svg, fadeout_strength=0, **kwargs)
        self.__check_svg__(svg, cell, **kwargs)

    def test_draw_simple_xyz(self):
        for s in (True, False):
            self.__test_0__(self.cell, show_cell=s,)
        for c in ('x', 'y', 'z'):
            self.__test_0__(self.cell, camera=c)
        for atoms in (True, False):
            self.__test_0__(self.cell, show_atoms=atoms)
        for bonds in (True, False):
            self.__test_0__(self.cell, show_bonds=bonds)

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
        svg = svgwrite_unit_cell(self.cell2, fl, show_cell=True, fadeout_strength=0)
        self.__check_svg__(svg, self.cell2)
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
        self.cell = Cell(
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
