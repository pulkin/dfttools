import math
import os
import unittest

import numericalunits
import numpy
from dfttools.parsers.generic import ParseError
from dfttools.parsers.qe import bands, output, cond, input, proj
from dfttools.types import Basis
from numpy import testing


class Test_bands0(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/qe.bands.0.testcase"), 'r') as f:
            self.parser = bands(f.read())

    def test_bands(self):
        c = Basis(
            numpy.array((
                (1.000000, -0.577350, 0.000000),
                (0.000000, 1.154701, 0.000000),
                (0.000000, 0.000000, 0.158745))
            ) * 2 * math.pi / 5.999694 / numericalunits.aBohr
        )
        c = self.parser.bands(c)
        assert c.values.shape == (400, 34)
        assert c.coordinates.shape == (400, 3)
        testing.assert_allclose(c.coordinates[0, :], (0.500000, -0.288675, 0.000000))
        testing.assert_allclose(c.values[0, :], numpy.array(
            (0.500, -0.500, 0.185, -0.185, 0.248, -0.248, 0.067, -0.067, -0.498, 0.498,
             -0.498, 0.498, -0.500, 0.500, -0.499, 0.499, 0.500, -0.500, 0.499, -0.499,
             -0.498, 0.498, -0.498, 0.498, 0.499, -0.499, 0.499, -0.499, 0.496, -0.496,
             0.496, -0.496, 0.499, -0.499)
        ) * numericalunits.eV)
        testing.assert_allclose(c.coordinates[-1, :], (-0.503333, -0.282902, 0.000000))
        testing.assert_allclose(c.values[-1, :], numpy.array(
            (0.500, -0.500, 0.185, -0.185, -0.250, 0.247, -0.065, 0.069, 0.497, -0.498,
             0.498, -0.497, -0.500, 0.500, 0.499, -0.499, 0.500, -0.500, -0.499, 0.499,
             -0.498, 0.498, -0.497, 0.498, 0.499, -0.499, 0.499, -0.499, -0.496, 0.496,
             0.496, -0.496, -0.499, 0.499)
        ) * numericalunits.eV)

    def test_valid_header(self):
        assert bands.valid_header(self.parser.parser.string[:1000])


class Test_output0(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/qe.output.0.testcase"), 'r') as f:
            self.parser = output(f.read())

    def test_success(self):
        assert self.parser.success()

    def test_routineError(self):
        assert self.parser.routineError() is None

    def test_scf_accuracy(self):
        a = self.parser.scf_accuracy()
        assert a.shape == (90,)
        assert a[0] == 0.01555592 * numericalunits.Ry
        assert a[-1] == 9.4e-9 * numericalunits.Ry

    def test_scf_steps(self):
        a = self.parser.scf_steps()
        assert a.shape == (19,)
        assert a[0] == 5
        assert a[-1] == 1
        assert a.sum() == 90

    def test_scf_failed(self):
        assert not self.parser.scf_failed()

    def test_fermi(self):
        a = self.parser.fermi()
        assert a.shape == (19,)
        assert a[0] == 10.0033 * numericalunits.eV
        assert a[-1] == 8.2525 * numericalunits.eV

    def test_total(self):
        a = self.parser.total()
        assert a.shape == (19,)
        assert a[0] == -25.44012222 * numericalunits.Ry
        assert a[-1] == -25.49951614 * numericalunits.Ry

    def test_threads(self):
        assert self.parser.threads() == 1

    def test_time(self):
        a = self.parser.time()
        assert a.shape == (self.parser.scf_accuracy().shape[0] + self.parser.scf_steps().shape[0],)
        assert a[0] == 0.4
        assert a[-1] == 27.2

    def test_alat(self):
        assert self.parser.alat() == 7.0103 * numericalunits.aBohr

    def test_unitCells(self):
        cells = self.parser.unitCells()
        assert len(cells) == 20
        for i in range(20):
            testing.assert_equal(cells[i].values,
                                 ('As', 'As')
                                 )
        assert cells[0].units_aware()
        testing.assert_allclose(cells[0].vectors, numpy.array(
            ((0.580130, 0.000000, 0.814524),
             (-0.290065, 0.502407, 0.814524),
             (-0.290065, -0.502407, 0.814524))
        ) * 7.0103 * numericalunits.aBohr)
        testing.assert_allclose(cells[0].reciprocal().vectors, numpy.array(
            ((1.149169, 0.000000, 0.409237),
             (-0.574584, 0.995209, 0.409237),
             (-0.574584, -0.995209, 0.409237))
        ) / 7.0103 / numericalunits.aBohr, rtol=1e-5, atol=1)
        testing.assert_allclose(cells[0].cartesian(), numpy.array(
            ((0.0000001, 0.0000000, 0.7086605),
             (-0.0000001, 0.0000000, -0.7086605))
        ) * 7.0103 * numericalunits.aBohr, atol=1)
        assert cells[1].units_aware()
        testing.assert_allclose(cells[1].vectors, numpy.array(
            ((0.589711141, -0.000000000, 0.822239221),
             (-0.294855381, 0.510704782, 0.822239223),
             (-0.294855381, -0.510704782, 0.822239223))
        ) * 7.01033623 * numericalunits.aBohr)
        testing.assert_allclose(cells[1].coordinates,
                                ((0.288386168, 0.288386167, 0.288386167),
                                 (-0.288386168, -0.288386167, -0.288386167))
                                )
        assert cells[-1].units_aware()
        testing.assert_allclose(cells[-1].vectors, numpy.array(
            ((0.593659483, -0.000000000, 0.870567646),
             (-0.296829546, 0.514124144, 0.870567651),
             (-0.296829546, -0.514124144, 0.870567651))
        ) * 7.01033623 * numericalunits.aBohr)
        testing.assert_allclose(cells[-1].coordinates,
                                ((0.272235154, 0.272235145, 0.272235145),
                                 (-0.272235154, -0.272235145, -0.272235145))
                                )

    def test_bands(self):
        self.assertRaises(Exception, self.parser.bands)
        b = self.parser.bands(index=None, skipVCRelaxException=True)
        assert len(b) == 19
        for i in range(len(b)):
            testing.assert_allclose(b[0].vectors, numpy.array(
                ((1.149169, 0.000000, 0.409237),
                 (-0.574584, 0.995209, 0.409237),
                 (-0.574584, -0.995209, 0.409237))
            ) * 2 * math.pi / 7.0103 / numericalunits.aBohr)

        testing.assert_allclose(b[0].cartesian(), numpy.array(
            ((0.0000, 0.0000, 0.1535),
             (-0.1436, -0.2488, 0.2558),
             (0.2873, 0.4976, -0.0512),
             (0.1436, 0.2488, 0.0512),
             (-0.2873, 0.0000, 0.3581),
             (0.1436, 0.7464, 0.0512),
             (0.0000, 0.4976, 0.1535),
             (0.5746, 0.0000, -0.2558),
             (0.0000, 0.0000, 0.4604),
             (0.4309, 0.7464, 0.1535))
        ) * 2 * math.pi / 7.0103 / numericalunits.aBohr, atol=1e-4, rtol=1)
        testing.assert_allclose(b[0].values, numpy.array(
            ((-6.9960, 4.5196, 5.9667, 5.9667, 8.4360, 11.0403, 11.7601, 11.7601, 16.5645),
             (-5.9250, 0.3917, 5.3512, 5.6501, 9.2996, 10.5303, 11.7005, 13.5632, 15.7167),
             (-4.3490, -2.4704, 4.7883, 6.1554, 7.8796, 10.8149, 12.5849, 13.8261, 17.7262),
             (-6.3695, 1.3043, 4.9860, 7.1720, 8.5435, 10.8049, 12.4702, 13.9612, 15.3511),
             (-5.5427, 1.1264, 3.5658, 4.2978, 7.5159, 10.4217, 13.7076, 13.7746, 16.9045),
             (-3.8393, -1.8099, 2.3270, 4.2466, 8.0539, 11.6204, 13.3234, 15.7202, 17.3489),
             (-4.7124, -1.4722, 3.0016, 6.6926, 7.7777, 12.3034, 13.0675, 13.4304, 16.0962),
             (-4.0542, -1.5061, 3.7084, 3.7296, 6.0243, 10.0593, 15.9112, 17.7151, 18.4776),
             (-5.8586, 0.8361, 5.8840, 5.8840, 7.4114, 10.0627, 10.0627, 12.1191, 17.3944),
             (-4.8492, -0.0498, 2.4338, 4.7831, 7.5088, 11.6828, 12.0642, 14.4760, 17.7700))
        ) * numericalunits.eV)
        testing.assert_allclose(b[-1].cartesian(), numpy.array(
            ((0.0000, -0.0000, 0.1436),
             (-0.1404, -0.2431, 0.2393),
             (0.2807, 0.4863, -0.0479),
             (0.1404, 0.2431, 0.0479),
             (-0.2807, 0.0000, 0.3350),
             (0.1404, 0.7294, 0.0479),
             (0.0000, 0.4863, 0.1436),
             (0.5615, -0.0000, -0.2393),
             (0.0000, 0.0000, 0.4308),
             (0.4211, 0.7294, 0.1436))
        ) * 2 * math.pi / 7.0103 / numericalunits.aBohr, atol=1e-4, rtol=1)
        testing.assert_allclose(b[-1].values, numpy.array(
            ((-7.1166, 1.7721, 5.6229, 5.6229, 6.5346, 9.9927, 10.5558, 10.5558, 14.5337),
             (-6.0954, -0.8486, 3.9924, 5.6835, 8.0573, 8.3099, 9.0569, 11.8901, 13.9362),
             (-4.5680, -3.1921, 4.5877, 4.7617, 6.2465, 9.3192, 9.6639, 10.4217, 15.6406),
             (-6.5374, 0.1870, 4.7448, 5.3190, 6.7072, 9.4267, 10.2382, 11.4742, 13.4682),
             (-5.7241, -0.6159, 2.9715, 4.0569, 5.3425, 10.2036, 11.9662, 12.0557, 13.7711),
             (-4.1437, -2.5617, 1.8718, 2.8655, 6.2034, 9.9254, 12.5118, 13.7278, 14.0308),
             (-4.9960, -2.1895, 2.8200, 4.7932, 6.1128, 9.4156, 11.1803, 12.2083, 13.7182),
             (-4.4645, -1.9020, 1.8730, 3.5230, 4.1472, 9.7977, 12.9771, 14.3143, 14.9436),
             (-5.9208, -1.5477, 5.7981, 5.7981, 7.0153, 8.5039, 8.5039, 9.6260, 15.7219),
             (-4.9074, -2.0659, 2.1277, 4.6402, 5.9533, 10.0685, 10.3976, 13.2007, 15.2380))
        ) * numericalunits.eV)
        assert b[0].meta["Fermi"] == 10.0033 * numericalunits.eV
        assert b[-1].meta["Fermi"] == 8.2525 * numericalunits.eV

        for i in range(19):
            bb = self.parser.bands(index=i, skipVCRelaxException=True)
            assert len(self.parser.parser.__history__) == 0
            assert bb == b[i]
            assert bb.meta["Fermi"] == b[i].meta["Fermi"]

        for i in range(19):
            bb = self.parser.bands(index=-i - 1, skipVCRelaxException=True)
            assert len(self.parser.parser.__history__) == 0
            assert bb == b[-i - 1]
            assert bb.meta["Fermi"] == b[-i - 1].meta["Fermi"]

        with self.assertRaises(ParseError):
            self.parser.bands(index=-21, skipVCRelaxException=True)

        with self.assertRaises(ParseError):
            self.parser.bands(index=19, skipVCRelaxException=True)

    def test_valid_header(self):
        assert output.valid_header(self.parser.parser.string[:1000])


class Test_output1(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/qe.output.1.testcase"), 'r') as f:
            self.parser = output(f.read())

    def test_unitCells(self):
        cells = self.parser.unitCells()
        assert len(cells) == 6
        for i in range(6):
            assert cells[i].units_aware()
            testing.assert_equal(cells[i].values,
                                 ('C', 'O')
                                 )
            testing.assert_allclose(cells[i].vectors, numpy.array(
                ((1.0, 0.0, 0.0),
                 (0.0, 1.0, 0.0),
                 (0.0, 0.0, 1.0))
            ) * 12.0 * numericalunits.aBohr)
        testing.assert_allclose(cells[0].cartesian(), numpy.array(
            ((0.1880000, 0.0000000, 0.0000000),
             (0.0000000, 0.0000000, 0.0000000))
        ) * 12 * numericalunits.aBohr)
        testing.assert_allclose(cells[1].cartesian(), numpy.array(
            ((2.040132676, 0.0, 0.0),
             (0.0, 0.0, 0.0))
        ) * numericalunits.aBohr)
        testing.assert_allclose(cells[-1].cartesian(), numpy.array(
            ((2.140073906, 0.0, 0.0),
             (0.0, 0.0, 0.0))
        ) * numericalunits.aBohr)


class Test_output2(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/qe.output.2.testcase"), 'r') as f:
            self.parser = output(f.read())

    def test_unitCells(self):
        cells = self.parser.unitCells()
        assert len(cells) == 14
        for i in range(14):
            assert cells[i].units_aware()
            testing.assert_equal(cells[i].values,
                                 ('Al',) * 7
                                 )
            testing.assert_allclose(cells[i].vectors, numpy.array(
                ((1.0, 0.0, 0.0),
                 (0.0, 1.0, 0.0),
                 (0.0, 0.0, 8.0))
            ) * 5.3033 * numericalunits.aBohr)
        testing.assert_allclose(cells[0].cartesian(), numpy.array(
            ((0.5000000, 0.5000000, -2.1213200),
             (0.0000000, 0.0000000, -1.4142130),
             (0.5000000, 0.5000000, -0.7071070),
             (0.0000000, 0.0000000, 0.0000000),
             (0.5000000, 0.5000000, 0.7071070),
             (0.0000000, 0.0000000, 1.4142130),
             (0.5000000, 0.5000000, 2.1213200))
        ) * 5.3033 * numericalunits.aBohr)
        testing.assert_allclose(cells[1].cartesian(), numpy.array(
            ((0.500000000, 0.500000000, -2.119426840),
             (0.000000000, 0.000000000, -1.414423364),
             (0.500000000, 0.500000000, -0.706630354),
             (0.000000000, 0.000000000, 0.000000000),
             (0.500000000, 0.500000000, 0.706630354),
             (0.000000000, 0.000000000, 1.414423364),
             (0.500000000, 0.500000000, 2.119426840))
        ) * 5.3033 * numericalunits.aBohr)
        testing.assert_allclose(cells[-1].cartesian(), numpy.array(
            ((0.500000000, 0.500000000, -2.062079273),
             (0.000000000, 0.000000000, -1.379793410),
             (0.500000000, 0.500000000, -0.689092900),
             (0.000000000, 0.000000000, 0.000000000),
             (0.500000000, 0.500000000, 0.689092900),
             (0.000000000, 0.000000000, 1.379793410),
             (0.500000000, 0.500000000, 2.062079273))
        ) * 5.3033 * numericalunits.aBohr)

    # def test_forces(self):
    # f = self.parser.forces()

    # assert f.shape[0] == 14

    # testing.assert_equal(f[0],numpy.array((
    # (0, 0,  0.01016766),
    # (0, 0, -0.00112981),
    # (0, 0,  0.00255994),
    # (0, 0,  0.00000000),
    # (0, 0, -0.00255994),
    # (0, 0,  0.00112981),
    # (0, 0, -0.01016766),
    # ))*numericalunits.Ry/numericalunits.aBohr)

    # testing.assert_equal(f[-1],numpy.array((
    # (0, 0, -0.00004768),
    # (0, 0,  0.00003912),
    # (0, 0,  0.00041843),
    # (0, 0,  0.00000000),
    # (0, 0, -0.00041843),
    # (0, 0, -0.00003912),
    # (0, 0,  0.00004768),
    # ))*numericalunits.Ry/numericalunits.aBohr)

    def test_force(self):
        f = self.parser.force()

        testing.assert_equal(f, numpy.array((
            0.014914,
            0.013792,
            0.013310,
            0.011289,
            0.007458,
            0.009847,
            0.005014,
            0.010593,
            0.002149,
            0.006964,
            0.000845,
            0.000598,
        )) * numericalunits.Ry / numericalunits.aBohr)


class Test_output3(unittest.TestCase):
    """
    Example copied from espresso-5.0.1/PW/examples/example01/results/si.band.cg.out
    """

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/qe.output.3.testcase"), 'r') as f:
            self.parser = output(f.read())

    def test_bands(self):
        b = self.parser.bands(index=None)
        assert len(b) == 1
        b = b[0]
        testing.assert_allclose(b.vectors, numpy.array(
            ((-1., -1., 1.),
             (1., 1., 1.),
             (-1., 1., -1.))
        ) * 2 * math.pi / 10.2 / numericalunits.aBohr)
        assert b.size() == 28
        crds = b.cartesian()
        testing.assert_allclose(crds[:, 0], numpy.array(
            (0.,) * 22 + \
            tuple(0.1 * i for i in range(6))
        ) * 2 * math.pi / 10.2 / numericalunits.aBohr, atol=1)
        testing.assert_allclose(crds[:, 1], numpy.array(
            (0.,) * 12 + \
            tuple(0.1 * i for i in range(1, 11)) + \
            tuple(0.1 * i for i in range(6))
        ) * 2 * math.pi / 10.2 / numericalunits.aBohr, atol=1)
        testing.assert_allclose(crds[:, 2], numpy.array(
            tuple(0.1 * i for i in range(11)) * 2 + \
            tuple(0.1 * i for i in range(6))
        ) * 2 * math.pi / 10.2 / numericalunits.aBohr, atol=1)
        testing.assert_allclose(b.values[0, :], numpy.array(
            (-5.8099, 6.2549, 6.2549, 6.2549, 8.8221, 8.8221, 8.8221, 9.7232)) * numericalunits.eV)
        testing.assert_allclose(b.values[-1, :], numpy.array(
            (-3.4180, -0.8220, 5.0289, 5.0289, 7.8139, 9.5968, 9.5968, 13.8378)) * numericalunits.eV)


class Test_output4(unittest.TestCase):
    """
    Fixed occupations (insulator).
    """

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/qe.output.4.testcase"), 'r') as f:
            self.parser = output(f.read())

    def test_fermi(self):
        f = self.parser.fermi()
        testing.assert_equal(f, numpy.array((
            0.1229,
            0.6136,
        )) * numericalunits.eV)

    def test_bands(self):
        b = self.parser.bands(index=None)

        testing.assert_equal(b[0].coordinates, (
            (0.0000000, 0.0000000, 0.0000000),
            (0.0000000, 0.0833333, 0.0000000),
            (0.0000000, 0.1666667, 0.0000000),
            (0.0000000, 0.2500000, 0.0000000),
            (0.0000000, 0.3333333, 0.0000000),
            (0.0000000, 0.4166667, 0.0000000),
            (0.0000000, -0.5000000, 0.0000000),
            (0.0833333, 0.1666667, 0.0000000),
            (0.0833333, 0.2500000, 0.0000000),
            (0.0833333, 0.3333333, 0.0000000),
            (0.0833333, 0.4166667, 0.0000000),
            (0.0833333, -0.5000000, 0.0000000),
            (0.1666667, 0.3333333, 0.0000000),
            (0.1666667, 0.4166667, 0.0000000),
            (0.1666667, -0.5000000, 0.0000000),
            (0.1666667, -0.4166667, 0.0000000),
            (0.2500000, -0.5000000, 0.0000000),
            (0.2500000, -0.4166667, 0.0000000),
            (0.3333333, -0.3333333, 0.0000000),
        ))


class Test_output5(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/qe.output.5.testcase"), 'r') as f:
            self.parser = output(f.read())

    def test_unitCells(self):
        cells = self.parser.unitCells()
        assert len(cells) == 4
        for i in range(4):
            testing.assert_equal(cells[i].values,
                                 ('W', 'Se', 'Se', 'W', 'Se', 'Se'),
                                 )
        testing.assert_allclose(cells[1].vectors, numpy.array((
            (5.952508067, 0, 0,),
            (0, 3.312746029, 0,),
            (0, 0, 19.999998523,),
        )) * numericalunits.angstrom)
        testing.assert_allclose(cells[1].coordinates, numpy.array((
            (0.125345331, 0.749999461, 0.494938933,),
            (0.359065123, 0.249999820, 0.570697112,),
            (0.510935861, 0.749999461, 0.429303122,),
            (0.744655908, 0.249999820, 0.505060896,),
            (0.852108020, 0.749999461, 0.594681796,),
            (0.017889757, 0.249999820, 0.405318141,),
        )))
        testing.assert_allclose(cells[-1].vectors, numpy.array((
            (5.952508067, 0, 0,),
            (0, 3.316388490, 0,),
            (0, 0, 19.999998523,),
        )) * numericalunits.angstrom)
        testing.assert_allclose(cells[-1].coordinates, numpy.array((
            (0.127475072, 0.749999461, 0.493768971,),
            (0.361369273, 0.249999820, 0.572048008,),
            (0.508630727, 0.749999461, 0.427951984,),
            (0.742525597, 0.249999820, 0.506231069,),
            (0.858847877, 0.749999461, 0.594463213,),
            (0.011151454, 0.249999820, 0.405536755,),
        )))


class Test_proj0(unittest.TestCase):
    """
    Projwfc run for example calculation from espresso-5.0.1/PW/examples/example01/results/si.scf.cg.in
    """

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/qe.proj.0.testcase"), 'r') as f:
            self.parser = proj(f.read())

    def test_basis(self):
        b = self.parser.basis()
        assert b.shape[0] == 8
        testing.assert_array_equal(b["state"], numpy.arange(1, 9))
        testing.assert_array_equal(b["atom"], (1,) * 4 + (2,) * 4)
        testing.assert_array_equal(b["name"], ("Si",) * 8)
        testing.assert_array_equal(b["wfc"], (1, 2, 2, 2, 1, 2, 2, 2))
        testing.assert_array_equal(b["l"], (0, 1, 1, 1, 0, 1, 1, 1))
        testing.assert_array_equal(b["m"], (1, 1, 2, 3, 1, 1, 2, 3))

    def test_weights(self):
        c = self.parser.weights()
        assert c.shape == (28, 8, 8)
        testing.assert_allclose(c[0, 0, numpy.array((0, 4))],
                                (0.498, 0.498)
                                )
        testing.assert_allclose(c[0, -1, numpy.array((0, 4))],
                                (0.489, 0.489)
                                )
        testing.assert_allclose(c[-1, 0, numpy.array((0, 4, 1, 2, 3, 5, 6, 7))],
                                (0.382, 0.382, 0.039, 0.039, 0.039, 0.039, 0.039, 0.039)
                                )
        testing.assert_allclose(c[-1, -1, numpy.array((1, 2, 3, 5, 6, 7, 0, 4))],
                                (0.023, 0.023, 0.023, 0.023, 0.023, 0.023, 0.020, 0.020)
                                )

    def test_weights_equality(self):
        c = self.parser.weights()
        c2 = self.parser._weights()
        testing.assert_allclose(c, c2)

    def test_weights_space(self):
        data = self.parser.data.replace('[# ', '[#').replace('[# ', '[#').replace('[# ', '[#')
        parser = proj(data)
        c = self.parser.weights()
        c2 = self.parser._weights()
        testing.assert_allclose(c, c2)

    def test_valid_header(self):
        assert proj.valid_header(self.parser.parser.string[:1000])


class Test_proj1(unittest.TestCase):
    """
    NC case.
    """

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/qe.proj.1.testcase"), 'r') as f:
            self.parser = proj(f.read())

    def test_basis(self):
        b = self.parser.basis()
        assert b.shape[0] == 42
        testing.assert_array_equal(b["state"], numpy.arange(1, 43))
        testing.assert_array_equal(b["atom"], (1,) * 26 + (2,) * 8 + (3,) * 8)
        testing.assert_array_equal(b["name"], ("Mo",) * 26 + ("S",) * 16)
        testing.assert_array_equal(b["wfc"], (
        1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 1, 1, 2, 2, 3, 3, 3, 3, 1, 1, 2,
        2, 3, 3, 3, 3))
        testing.assert_array_equal(b["j"], (.5,) * 6 + (1.5,) * 4 + (.5,) * 2 + (1.5,) * 8 + (2.5,) * 6 + (.5,) * 4 + (
        1.5,) * 4 + (.5,) * 4 + (1.5,) * 4)
        testing.assert_array_equal(b["l"], (0,) * 4 + (1,) * 12 + (2,) * 10 + (0,) * 2 + (1,) * 6 + (0,) * 2 + (1,) * 6)
        m1 = (-.5, .5)
        m2 = (-1.5, -.5, .5, 1.5)
        m3 = (-2.5, -1.5, -.5, .5, 1.5, 2.5)
        testing.assert_array_equal(b["m_j"], m1 * 3 + m2 + m1 + m2 * 2 + m3 + m1 * 2 + m2 + m1 * 2 + m2)


class Test_cond0(unittest.TestCase):
    """
    Standard case with left and right leads.
    """

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/qe.cond.0.testcase"), 'r') as f:
            self.parser = cond(f.read())

    def test_transmission(self):
        t = self.parser.transmission()
        assert t.shape == (20,)
        testing.assert_array_equal(t["energy"], (1.0574600,) * 20)
        testing.assert_array_equal(t["kx"], (0.,) * 20)
        testing.assert_array_equal(t["ky"],
                                   (0.3416214,) * 4 + (0.3411815,) * 4 + (0.3407045,) * 4 + (0.3400998,) * 4 + (
                                   0.3394089,) * 4
                                   )
        testing.assert_allclose(t["incoming"],
                                (0.0058519 + 0.0000342j,) * 2 + (0.0163700 + 0.0000356j,) * 2 + (
                                0.0074525 + 0.0000342j,) * 2 + \
                                (0.0170005 + 0.0000356j,) * 2 + (0.0087867 + 0.0000343j,) * 2 + (
                                0.0176185 + 0.0000356j,) * 2 + \
                                (0.0101270 + 0.0000343j,) * 2 + (0.0183148 + 0.0000355j,) * 2 + (
                                0.0113194 + 0.0000343j,) * 2 + \
                                (0.0189920 + 0.0000355j,) * 2
                                )
        testing.assert_allclose(t["outgoing"],
                                (0.0219061 - 0.0000346j, 0.0257629 - 0.0000359j) * 2 + \
                                (0.0223732 - 0.0000346j, 0.0261584 - 0.0000359j) * 2 + \
                                (0.0228378 - 0.0000347j, 0.0265540 - 0.0000359j) * 2 + \
                                (0.0233683 - 0.0000347j, 0.0270082 - 0.0000359j) * 2 + \
                                (0.0238892 - 0.0000347j, 0.0274576 - 0.0000359j) * 2
                                )

    def test_transmissionStates(self):
        t_in = self.parser.transmission(kind="states_in")
        t_out = self.parser.transmission(kind="states_out")
        assert t_in.shape == (10,)
        testing.assert_array_equal(t_in["energy"], (1.0574600,) * 10)
        testing.assert_array_equal(t_in["kx"], (0.,) * 10)
        testing.assert_array_equal(t_in["ky"],
                                   (0.3416214,) * 2 + (0.3411815,) * 2 + (0.3407045,) * 2 + (0.3400998,) * 2 + (
                                   0.3394089,) * 2
                                   )
        testing.assert_allclose(t_in["incoming"],
                                (0.0058519 + 0.0000342j, 0.0163700 + 0.0000356j,
                                 0.0074525 + 0.0000342j, 0.0170005 + 0.0000356j,
                                 0.0087867 + 0.0000343j, 0.0176185 + 0.0000356j,
                                 0.0101270 + 0.0000343j, 0.0183148 + 0.0000355j,
                                 0.0113194 + 0.0000343j, 0.0189920 + 0.0000355j)
                                )
        assert t_out.shape == (10,)
        testing.assert_array_equal(t_out["energy"], (1.0574600,) * 10)
        testing.assert_array_equal(t_out["kx"], (0.,) * 10)
        testing.assert_array_equal(t_out["ky"],
                                   (0.3416214,) * 2 + (0.3411815,) * 2 + (0.3407045,) * 2 + (0.3400998,) * 2 + (
                                   0.3394089,) * 2
                                   )
        testing.assert_allclose(t_out["outgoing"],
                                (0.0219061 - 0.0000346j, 0.0257629 - 0.0000359j,
                                 0.0223732 - 0.0000346j, 0.0261584 - 0.0000359j,
                                 0.0228378 - 0.0000347j, 0.0265540 - 0.0000359j,
                                 0.0233683 - 0.0000347j, 0.0270082 - 0.0000359j,
                                 0.0238892 - 0.0000347j, 0.0274576 - 0.0000359j)
                                )

    def test_valid_header(self):
        assert cond.valid_header(self.parser.parser.string[:1000])


class Test_cond1(unittest.TestCase):
    """
    Case with only left lead.
    """

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/qe.cond.1.testcase"), 'r') as f:
            self.parser = cond(f.read())

    def test_transmission(self):
        t = self.parser.transmission()
        assert t.shape == (20,)
        testing.assert_array_equal(t["energy"], (0.8838,) * 20)
        testing.assert_array_equal(t["kx"], (0.,) * 20)
        testing.assert_array_equal(t["ky"],
                                   (0.3148603,) * 4 + (0.3276829,) * 4 + (0.3420579,) * 4 + (0.3540834,) * 4 + (
                                   0.3065240,) * 4
                                   )
        testing.assert_allclose(t["incoming"],
                                (0.0359251,) * 2 + (0.0362036,) * 2 + (0.0472180,) * 2 + \
                                (0.0474508,) * 2 + (0.0466185,) * 2 + (0.0469464,) * 2 + \
                                (0.0331053,) * 2 + (0.0335629,) * 2 + (0.0154268,) * 2 + \
                                (0.0162679,) * 2
                                )
        testing.assert_allclose(t["outgoing"],
                                (0.0359251, 0.0362036) * 2 + \
                                (0.0472180, 0.0474508) * 2 + \
                                (0.0466185, 0.0469464) * 2 + \
                                (0.0331053, 0.0335629) * 2 + \
                                (0.0154268, 0.0162679) * 2
                                )


class Test_cond2(unittest.TestCase):
    """
    Unfinished calculation.
    """

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/qe.cond.2.testcase"), 'r') as f:
            self.parser = cond(f.read())

    def test_transmission(self):
        t = self.parser.transmission()
        assert t.shape == (8,)
        testing.assert_array_equal(t["energy"], (-0.8,) * 8)
        testing.assert_array_equal(t["kx"], (0.,) * 8)
        testing.assert_array_equal(t["ky"],
                                   (0.0661144,) * 4 + (0.0746435,) * 4
                                   )
        testing.assert_allclose(t["incoming"],
                                (-0.1231402 - 0.0000591j,) * 2 + (-0.1231402 - 0.0000591j,) * 2 + \
                                (-0.1086609 - 0.0000637j,) * 2 + (-0.1086609 - 0.0000637j,) * 2
                                )
        testing.assert_allclose(t["outgoing"],
                                (-0.1158360 + 0.0000655j, -0.1158360 + 0.0000655j) * 2 + \
                                (-0.0991168 + 0.0000653j, -0.0991168 + 0.0000653j) * 2
                                )
        testing.assert_allclose(t["transmission"],
                                (0, 0.00007, 0.00007, 0, 0.00001, 0.00004, 0.00004, 0.00001)
                                )

    def test_totalTransmission(self):
        t = self.parser.transmission(kind="total")
        assert t.shape == (4,)
        testing.assert_array_equal(t["energy"], (-0.8,) * 4)
        testing.assert_array_equal(t["kx"], (0.,) * 4)
        testing.assert_array_equal(t["ky"],
                                   (0.0661144,) * 2 + (0.0746435,) * 2
                                   )
        testing.assert_allclose(t["incoming"],
                                (-0.1231402 - 0.0000591j, -0.1231402 - 0.0000591j,
                                 -0.1086609 - 0.0000637j, -0.1086609 - 0.0000637j)
                                )
        testing.assert_allclose(t["transmission"],
                                (0.00007, 0.00007, 0.00005, 0.00005)
                                )


class Test_cond3(unittest.TestCase):
    """
    Test with different leads and lack of states in one of the leads.
    """

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/qe.cond.3.testcase"), 'r') as f:
            self.parser = cond(f.read())

    def test_transmission(self):
        t = self.parser.transmission()
        assert t.shape == (12,)
        testing.assert_array_equal(t["energy"], (-0.7611,) * 12)
        testing.assert_array_equal(t["kx"], (0.,) * 12)
        testing.assert_array_equal(t["ky"],
                                   (0.0249939,) * 4 + (0.0166581,) * 4 + (0.0079982,) * 4
                                   )
        testing.assert_allclose(t["incoming"],
                                (-0.0623236 - 0.0000757j,) * 4 + (-0.0698768 - 0.0000742j,) * 4 + \
                                (-0.0743667 - 0.0000752j,) * 4
                                )
        testing.assert_allclose(t["outgoing"],
                                (-0.0356423 + 0.0000767j,) * 4 + (-0.0480292 + 0.0000787j,) * 4 + \
                                (-0.0541478 + 0.0000784j,) * 4
                                )
        testing.assert_allclose(t["transmission"],
                                (0.00003, 0.00001, 0.00001, 0.00003,
                                 0.00005, 0.00001, 0.00001, 0.00005,
                                 0.00006, 0.00001, 0.00001, 0.00006)
                                )

    def test_totalTransmission(self):
        t = self.parser.transmission(kind="total")
        assert t.shape == (10,)
        testing.assert_array_equal(t["energy"], (-0.7611,) * 10)
        testing.assert_array_equal(t["kx"], (0.,) * 10)
        testing.assert_array_equal(t["ky"],
                                   (0.0399059,) * 2 + (0.0327777,) * 2 + (0.0249939,) * 2 + \
                                   (0.0166581,) * 2 + (0.0079982,) * 2
                                   )
        testing.assert_allclose(t["incoming"],
                                (-0.0321023 - 0.0000760j,) * 2 + (-0.0505708 - 0.0000759j,) * 2 + \
                                (-0.0623236 - 0.0000757j,) * 2 + (-0.0698768 - 0.0000742j,) * 2 + \
                                (-0.0743667 - 0.0000752j,) * 2
                                )
        testing.assert_allclose(t["transmission"],
                                (0, 0, 0, 0, 0.00003, 0.00003, 0.00005, 0.00005, 0.00007, 0.00007)
                                )


class Test_cond4(unittest.TestCase):
    """
    Test with different leads and different number of incoming and
    outgoing states, also with lack of incoming states.
    """

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cases/qe.cond.4.testcase"), 'r') as f:
            self.parser = cond(f.read())

    def test_transmission(self):
        t = self.parser.transmission()
        assert t.shape == (36,)
        testing.assert_array_equal(t["energy"], (1.1259,) * 24 + (1.0259,) * 12)
        testing.assert_array_equal(t["kx"], (0.,) * 36)
        testing.assert_array_equal(t["ky"],
                                   (0.3333333,) * 9 + (0.3233333,) * 6 + (0.3433333,) * 9 + \
                                   (0.3333333,) * 4 + (0.3233333,) * 4 + (0.3433333,) * 4
                                   )
        testing.assert_allclose(t["incoming"],
                                (0.0425931 - 0.0000019j,) * 3 + (0.1171181 + 0.0000037j,) * 3 + \
                                (0.1242032 + 0.0000042j,) * 3 + (0.0153953 - 0.0000025j,) * 2 + \
                                (0.1149962 + 0.0000040j,) * 2 + (0.1219607 + 0.0000045j,) * 2 + \
                                (0.0487266 - 0.0000013j,) * 3 + (0.1166978 + 0.0000035j,) * 3 + \
                                (0.1240200 + 0.0000039j,) * 3 + (0.0528382 + 0.0000034j,) * 2 + \
                                (0.0542379 + 0.0000040j,) * 2 + (0.0501590 + 0.0000036j,) * 2 + \
                                (0.0516561 + 0.0000042j,) * 2 + (0.0496596 + 0.0000033j,) * 2 + \
                                (0.0510848 + 0.0000038j,) * 2
                                )
        testing.assert_allclose(t["outgoing"],
                                (0.0391023 - 0.0000006j, 0.1161826 - 0.0000026j, 0.1232164 - 0.0000025j) * 3 + \
                                (0.1140708 - 0.0000026j, 0.1209898 - 0.0000026j) * 3 + \
                                (0.0457229 - 0.0000004j, 0.1157312 - 0.0000026j, 0.1229968 - 0.0000024j) * 3 + \
                                (0.0510829 - 0.0000028j, 0.0523475 - 0.0000031j) * 2 + \
                                (0.0483604 - 0.0000028j, 0.0497307 - 0.0000031j) * 2 + \
                                (0.0477319 - 0.0000028j, 0.0490073 - 0.0000031j) * 2
                                )
        testing.assert_allclose(t["transmission"],
                                (0.00000, 0.00006, 0.00000,
                                 0.00006, 0.00000, 0.00234,
                                 0.00000, 0.00234, 0.00000,
                                 0.00002, 0.00000,
                                 0.00000, 0.00214,
                                 0.00214, 0.00000,
                                 0.00000, 0.00007, 0.00000,
                                 0.00007, 0.00000, 0.00240,
                                 0.00000, 0.00240, 0.00000,
                                 0.00000, 0.00047,
                                 0.00047, 0.00000,
                                 0.00000, 0.00040,
                                 0.00040, 0.00000,
                                 0.00000, 0.00044,
                                 0.00044, 0.00000)
                                )

    def test_totalTransmission(self):
        t = self.parser.transmission(kind="total")
        assert t.shape == (15,)
        testing.assert_array_equal(t["energy"], (1.1259,) * 9 + (1.0259,) * 6)
        testing.assert_array_equal(t["kx"], (0.,) * 15)
        testing.assert_array_equal(t["ky"],
                                   (0.3333333,) * 3 + (0.3233333,) * 3 + (0.3433333,) * 3 + \
                                   (0.3333333,) * 2 + (0.3233333,) * 2 + (0.3433333,) * 2
                                   )
        testing.assert_allclose(t["incoming"],
                                (0.0425931 - 0.0000019j, 0.1171181 + 0.0000037j, 0.1242032 + 0.0000042j,
                                 0.0153953 - 0.0000025j, 0.1149962 + 0.0000040j, 0.1219607 + 0.0000045j,
                                 0.0487266 - 0.0000013j, 0.1166978 + 0.0000035j, 0.1240200 + 0.0000039j,
                                 0.0528382 + 0.0000034j, 0.0542379 + 0.0000040j,
                                 0.0501590 + 0.0000036j, 0.0516561 + 0.0000042j,
                                 0.0496596 + 0.0000033j, 0.0510848 + 0.0000038j)
                                )
        testing.assert_allclose(t["transmission"],
                                (0.00006, 0.00239, 0.00234,
                                 0.00002, 0.00214, 0.00214,
                                 0.00007, 0.00247, 0.00240,
                                 0.00047, 0.00047,
                                 0.00040, 0.00040,
                                 0.00044, 0.00044)
                                )


class Test_input0(unittest.TestCase):
    def setUp(self):
        raw = """
&control
    calculation = 'scf'
    outdir = './out'
    prefix = 'leads'
    restart_mode = 'from_scratch'
    verbosity = 'high'
    wf_collect = .true.
/
&system
    celldm(1) = 5.669178374
    celldm(2) = 1
    celldm(3) = 0.9466666667
    celldm(4) = 0
    celldm(5) = 0
    celldm(6) = 0
    degauss = 0.001
    ecutrho = 300
    ecutwfc = 50
    ibrav = 14
    lspinorb = .true.
    nat = 2
    noncolin = .true.
    ntyp = 1
    occupations = 'smearing'
/
&electrons
    electron_maxstep = 500
    mixing_beta = 0.2
/
&ions
/
ATOMIC_SPECIES
    c 1.000000 C.pbe-rrkjus.UPF
ATOMIC_POSITIONS crystal
    c 0.500000 0.500000 0.300000 1 1 1
    c 0.500000 0.500000 0.700000 1 1 1
K_POINTS automatic
    1, 1, 32, 0, 0, 0
ATOMIC_SPECIES
    c 1.000000 C.pbe-rrkjus.UPF
ATOMIC_POSITIONS crystal
    c 0.500000 0.500000 0.300000 1 1 1
    c 0.500000 0.500000 0.700000 1 1 1
K_POINTS automatic
    1, 1, 32, 0, 0, 0
"""
        self.parser = input(raw)

    def test_unitCell(self):
        cell = self.parser.unitCell()
        assert cell.units_aware()

        testing.assert_allclose(cell.vectors,
                                ((5.669178374 * numericalunits.aBohr, 0, 0),
                                 (0, 5.669178374 * numericalunits.aBohr, 0),
                                 (0, 0, 5.669178374 * numericalunits.aBohr * 0.9466666667)),
                                rtol=1e-10,
                                atol=1e-15,
                                )
        testing.assert_array_equal(cell.coordinates,
                                   ((.5, .5, .3), (.5, .5, .7))
                                   )
        testing.assert_array_equal(cell.values,
                                   ('c',) * 2
                                   )

    def test_valid_header(self):
        assert input.valid_header(self.parser.parser.string[:1000])
