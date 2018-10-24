import math
import unittest

import matplotlib
import numericalunits
import numpy
from numericalunits import eV, Ry, angstrom
from numpy import testing

matplotlib.use('SVG')
from matplotlib import pyplot
from matplotlib.testing.decorators import cleanup
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.image import AxesImage

from dfttools.types import Basis, UnitCell, Grid
from dfttools.presentation import matplotlib_bands, matplotlib_scalar, matplotlib_bands_density


class BandPlotTest(unittest.TestCase):

    def setUp(self):
        basis = Basis((1, 1, 1, 0, 0, -0.5), kind='triclinic')

        kp_gamma = numpy.array((0, 0, 0))[numpy.newaxis, :]
        kp_m = numpy.array((0.5, 0.0, 0))[numpy.newaxis, :]
        kp_k = numpy.array((2. / 3, 1. / 3, 0))[numpy.newaxis, :]

        kp_path = numpy.linspace(0, 1, 30)[:, numpy.newaxis]

        kp_path = numpy.concatenate((
            kp_gamma * (1 - kp_path) + kp_m * kp_path,
            kp_m * (1 - kp_path) + kp_k * kp_path,
            kp_k * (1 - kp_path) + kp_gamma * kp_path,
        ), axis=0)

        d = (basis.transform_to_cartesian(kp_path) ** 2).sum(axis=1)

        self.bands = UnitCell(
            basis,
            kp_path,
            ([[0, 0, 3]] + d[..., numpy.newaxis] * [[1, 2, -3]]) * eV,
        )
        self.bands.meta["Fermi"] = 1 * eV
        self.weights = self.bands.values / self.bands.values.max()
        self.huge_bands = UnitCell(
            self.bands,
            kp_path,
            ([BandPlotTest.__pseudo_random__(0, 1000, 50) * 10 - 5] + d[..., numpy.newaxis] * [
                BandPlotTest.__pseudo_random__(1000, 2000, 50) * 20 - 10]) * eV,
        )

    @staticmethod
    def __pseudo_random__(fr, to, size):
        return numpy.modf(numpy.linspace(fr, to, size) * math.pi)[0]

    def _test_plot(self, units):
        rlc = matplotlib_bands(self.bands, pyplot.gca(), energy_units=units)

        axes = pyplot.gcf().axes
        assert len(axes) == 1
        axes = axes[0]

        lc = axes.findobj(LineCollection)
        assert len(lc) == 1
        lc = lc[0]
        assert lc == rlc

        segments = lc.get_segments()
        assert len(segments) == (30 * 3 - 3) * 3

        testing.assert_allclose(axes.get_ylim(), numpy.array((-0.15, 3.15)) * eV / getattr(numericalunits, units))
        assert axes.get_yaxis().get_label().get_text().endswith("(" + units + ")")

    @cleanup
    def test_plot_eV(self):
        self._test_plot("eV")

    @cleanup
    def test_plot_Ry(self):
        self._test_plot("Ry")

    @cleanup
    def test_custom_units(self):
        matplotlib_bands(self.bands, pyplot.gca(), energy_units=2 * Ry, energy_units_name="Hartree")
        assert pyplot.gca().get_yaxis().get_label().get_text().endswith("(Hartree)")

    @cleanup
    def test_unknown_units(self):
        matplotlib_bands(self.bands, pyplot.gca(), energy_units=2 * Ry)
        assert pyplot.gca().get_yaxis().get_label().get_text() == 'Energy'

    @cleanup
    def test_plot_weights_color(self):
        rlc = matplotlib_bands(self.bands, pyplot.gca(), weights_color=self.weights)
        a = rlc.get_array()
        b = self.weights
        assert a[0] == 0.5 * (b[0, 0] + b[1, 0])
        assert a[-1] == 0.5 * (b[-2, -1] + b[-1, -1])

    @cleanup
    def test_plot_weights_color_error(self):
        with self.assertRaises(TypeError):
            matplotlib_bands(self.bands, pyplot.gca(), weights_color=self.weights[..., numpy.newaxis])

    @cleanup
    def test_plot_weights_size(self):
        rlc = matplotlib_bands(self.bands, pyplot.gca(), weights_size=self.weights)
        a = rlc.get_linewidth()
        b = self.weights
        assert a[0] == 0.5 * (b[0, 0] + b[1, 0])
        assert a[-1] == 0.5 * (b[-2, -1] + b[-1, -1])

    @cleanup
    def test_plot_weights_size_error(self):
        with self.assertRaises(TypeError):
            matplotlib_bands(self.bands, pyplot.gca(), weights_size=self.weights[..., numpy.newaxis])

    @cleanup
    def test_plot_weights_error(self):
        with self.assertRaises(TypeError):
            matplotlib_bands(self.bands, pyplot.gca(), weights=self.weights[..., numpy.newaxis])

    @cleanup
    def test_huge_bands(self):
        matplotlib_bands(self.huge_bands, pyplot.gca())
        axes = pyplot.gca()
        y = axes.get_ylim()
        assert y[0] > -2 and y[0] < -1
        assert y[1] > 3 and y[1] < 4


class BandDensityPlotTest(unittest.TestCase):

    def setUp(self):
        basis = Basis((1, 1, 1, 0, 0, -0.5), kind='triclinic', meta={"Fermi": 0})

        kp_gamma = numpy.array((0, 0, 0))[numpy.newaxis, :]
        kp_m = numpy.array((0.5, 0.0, 0))[numpy.newaxis, :]
        kp_k = numpy.array((2. / 3, 1. / 3, 0))[numpy.newaxis, :]

        kp_path = numpy.linspace(0, 1, 30)[:, numpy.newaxis]

        kp_path = numpy.concatenate((
            kp_gamma * (1 - kp_path) + kp_m * kp_path,
            kp_m * (1 - kp_path) + kp_k * kp_path,
            kp_k * (1 - kp_path) + kp_gamma * kp_path,
        ), axis=0)

        k = basis.transform_to_cartesian(kp_path) * math.pi / 3. ** .5 * 2
        e = (1 + 4 * numpy.cos(k[..., 1]) ** 2 + 4 * numpy.cos(k[..., 1]) * numpy.cos(k[..., 0] * 3. ** .5)) ** .5

        self.cell = UnitCell(
            basis,
            kp_path,
            e[:, numpy.newaxis] * eV * [[-1., 1.]],
        )
        self.cell_weights = self.cell.values / self.cell.values.max()

        self.grid = Grid(
            basis,
            (numpy.linspace(0, 1, 30, endpoint=False) + 1. / 60, numpy.linspace(0, 1, 30, endpoint=False) + 1. / 60,
             (0,)),
            numpy.zeros((30, 30, 1, 2), dtype=numpy.float64),
        )
        k = self.grid.cartesian() * math.pi / 3. ** .5 * 2
        e = (1 + 4 * numpy.cos(k[..., 1]) ** 2 + 4 * numpy.cos(k[..., 1]) * numpy.cos(k[..., 0] * 3. ** .5)) ** .5 * eV

        self.grid.values[..., 0] = -e
        self.grid.values[..., 1] = e

    @cleanup
    def test_plot(self):
        rl = matplotlib_bands_density(self.cell, pyplot.gca(), 100, show_fermi=False)
        assert len(rl) == 1
        rl = rl[0]

        axes = pyplot.gcf().axes
        assert len(axes) == 1
        axes = axes[0]

        testing.assert_allclose(axes.get_xlim(), (-3.3, 3.3))

        l = list(i for i in axes.get_children() if isinstance(i, Line2D))
        assert len(l) == 1
        l = l[0]
        assert l == rl

        x, y = l.get_data()

        testing.assert_allclose(x, numpy.linspace(-3.3, 3.3, 100))
        assert numpy.all(y < 10) and numpy.all(y > 0)
        assert numpy.any(y > 0.1)

    @cleanup
    def test_plot_weights(self):
        w1 = 0.3 * numpy.ones(self.cell.values.shape)
        w2 = 0.7 * numpy.ones(self.cell.values.shape)

        rl1 = matplotlib_bands_density(self.cell, pyplot.gca(), 100, show_fermi=False, weights=w1)[0]
        rl2 = matplotlib_bands_density(self.cell, pyplot.gca(), 100, show_fermi=False, weights=w2, on_top_of=w1)[0]

        x1, y1 = rl1.get_data()
        x2, y2 = rl2.get_data()

        testing.assert_allclose(x1, numpy.linspace(-3.3, 3.3, 100))
        testing.assert_allclose(x2, numpy.linspace(-3.3, 3.3, 100))
        testing.assert_allclose(y1, 0.3 * y2)

    @cleanup
    def test_plot_fill(self):
        rpc = matplotlib_bands_density(self.cell, pyplot.gca(), 100, show_fermi=False, use_fill=True)

        axes = pyplot.gcf().axes
        assert len(axes) == 1
        axes = axes[0]

        pc = list(i for i in axes.get_children() if isinstance(i, PolyCollection))
        assert len(pc) == 1
        pc = pc[0]
        assert pc == rpc

        for p in rpc.get_paths():
            (xmin, ymin), (xmax, ymax) = p.get_extents().get_points()
            assert xmin >= -3.31
            assert ymin >= 0
            assert xmax <= 3.31
            assert ymax < 10

    @cleanup
    def test_plot_fill_weights(self):
        w1 = 0.3 * numpy.ones(self.cell.values.shape)
        w2 = 0.7 * numpy.ones(self.cell.values.shape)

        rpc1 = matplotlib_bands_density(self.cell, pyplot.gca(), 100, show_fermi=False, weights=w1, use_fill=True)
        rpc2 = matplotlib_bands_density(self.cell, pyplot.gca(), 100, show_fermi=False, weights=w2, on_top_of=w1,
                                        use_fill=True)

        for rpc in (rpc1, rpc2):
            for p in rpc.get_paths():
                (xmin, ymin), (xmax, ymax) = p.get_extents().get_points()
                assert xmin >= -3.31
                assert ymin >= 0
                assert xmax <= 3.31
                assert ymax < 10

    @cleanup
    def test_plot_fill_portrait(self):
        rpc = matplotlib_bands_density(self.cell, pyplot.gca(), 100, show_fermi=False, use_fill=True,
                                       orientation="portrait")

        for p in rpc.get_paths():
            (xmin, ymin), (xmax, ymax) = p.get_extents().get_points()
            assert ymin >= -3.31
            assert xmin >= 0
            assert ymax <= 3.31
            assert xmax < 10

    @cleanup
    def test_units(self):
        matplotlib_bands_density(self.cell, pyplot.gca(), 100, units="eV")

        assert pyplot.gca().get_xaxis().get_label().get_text().endswith("(eV)")
        assert pyplot.gca().get_yaxis().get_label().get_text().endswith("(bands per eV)")

    @cleanup
    def test_custom_units_landscape(self):
        matplotlib_bands_density(self.cell, pyplot.gca(), 100, units=2 * Ry, units_name="Hartree",
                                 orientation='landscape')

        assert pyplot.gca().get_xaxis().get_label().get_text().endswith("(Hartree)")
        assert pyplot.gca().get_yaxis().get_label().get_text().endswith("(bands per Hartree)")

    @cleanup
    def test_custom_units_portrait(self):
        matplotlib_bands_density(self.cell, pyplot.gca(), 100, units=2 * Ry, units_name="Hartree",
                                 orientation='portrait')

        assert pyplot.gca().get_xaxis().get_label().get_text().endswith("(bands per Hartree)")
        assert pyplot.gca().get_yaxis().get_label().get_text().endswith("(Hartree)")

    @cleanup
    def test_unknown_units_landscape(self):
        matplotlib_bands_density(self.grid, pyplot.gca(), 100, units=2 * Ry, orientation='landscape')

        assert pyplot.gca().get_xaxis().get_label().get_text() == 'Energy'
        assert pyplot.gca().get_yaxis().get_label().get_text() == 'Density'

    @cleanup
    def test_unknown_units_portrait(self):
        matplotlib_bands_density(self.grid, pyplot.gca(), 100, units=2 * Ry, orientation='portrait')

        assert pyplot.gca().get_yaxis().get_label().get_text() == 'Energy'
        assert pyplot.gca().get_xaxis().get_label().get_text() == 'Density'

    @cleanup
    def test_portrait(self):
        rl = matplotlib_bands_density(self.cell, pyplot.gca(), 100, energy_range=(-3, 3), orientation="portrait")[0]

        l = list(i for i in pyplot.gca().get_children() if isinstance(i, Line2D))
        assert len(l) == 2

        x, y = rl.get_data()

        testing.assert_equal(y, numpy.linspace(-3, 3, 100))
        assert numpy.all(x < 10) and numpy.all(x > 0)
        assert numpy.any(x > 0.1)

    @cleanup
    def test_plot_grid(self):
        matplotlib_bands_density(self.grid, pyplot.gca(), 100, energy_range=(-3, 3), show_fermi=False)
        matplotlib_bands_density(self.grid, pyplot.gca(), 100, energy_range=(-3, 3), method='gaussian',
                                 show_fermi=False)

        l = list(i for i in pyplot.gca().get_children() if isinstance(i, Line2D))
        assert len(l) == 2

        yy = []

        for ll in l:
            x, y = ll.get_data()
            yy.append(y)

            testing.assert_equal(x, numpy.linspace(-3, 3, 100))
            assert numpy.all(y < 10) and numpy.all(y >= 0)
            assert numpy.any(y > 0.1)

        assert (numpy.abs(yy[0] - yy[1]) ** 2).sum() ** .5 / 100 < 1e-2

    @cleanup
    def test_unknown_orientation(self):
        with self.assertRaises(ValueError):
            matplotlib_bands_density(self.cell, pyplot.gca(), 100, energy_range=(-3, 3), orientation="unknown")


class ScalarGridPlotTest(unittest.TestCase):

    def setUp(self):
        self.grid = Grid(
            Basis((1 * angstrom, 1 * angstrom, 1 * angstrom, 0, 0, -0.5), kind='triclinic'),
            (
                numpy.linspace(0, 1, 30, endpoint=False),
                numpy.linspace(0, 1, 30, endpoint=False),
                numpy.linspace(0, 1, 30, endpoint=False),
            ),
            numpy.zeros((30, 30, 30)),
        )
        self.grid.values = numpy.prod(numpy.sin(self.grid.explicit_coordinates() * 2 * math.pi), axis=-1)

        self.wrong_dims = Grid(
            Basis(((1 * angstrom, 0), (0, 1 * angstrom))),
            (
                numpy.linspace(0, 1, 30, endpoint=False),
                numpy.linspace(0, 1, 30, endpoint=False),
            ),
            numpy.zeros((30, 30)),
        )

    @cleanup
    def test_plot(self):
        im = matplotlib_scalar(self.grid, pyplot.gca(), (0.1, 0.1, 0.1), 'z')

        axes = pyplot.gcf().axes
        assert len(axes) == 1
        axes = axes[0]

        ai = axes.findobj(AxesImage)
        assert len(ai) == 1
        ai = ai[0]
        assert ai == im

        testing.assert_allclose(axes.get_xlim(), (-0.55, 0.95))
        testing.assert_allclose(axes.get_ylim(), (-0.1 * 3. ** .5 / 2, 0.9 * 3. ** .5 / 2))

    @cleanup
    def test_plot_2(self):
        im = matplotlib_scalar(self.grid, pyplot.gca(), (0.1, 0.1, 0.1), 'z', ppu=10, margins=0)
        testing.assert_equal(im.get_size(), (round(10 * 3. ** .5 / 2), 15))

    @cleanup
    def test_plot_3(self):
        im = matplotlib_scalar(self.grid, pyplot.gca(), (0.1, 0.1, 0.1), 'z', show_cell=True)
        l = list(i for i in pyplot.gca().get_children() if isinstance(i, Line2D))
        assert len(l) == 12

    @cleanup
    def test_plot_error_1(self):
        with self.assertRaises(TypeError):
            matplotlib_scalar(self.wrong_dims, pyplot.gca(), (0.1, 0.1, 0.1), 'z')
