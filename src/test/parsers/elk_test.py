import unittest
import os
import math

from numpy import testing
import numpy
import numericalunits

from dfttools.parsers.elk import input, output, bands, unitcells
from dfttools.types import UnitCell, Basis

class Test_input0(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/elk.input.0.testcase"),'r') as f:
            self.parser = input(f.read())
            
    def test_unitCell(self):
        c = self.parser.unitCell()
        
        testing.assert_equal(c.vectors, numpy.array((
            (11.05491144, 0.000000000,-0.3046858492),
            (0.000000000, 6.194778086,  0.000000000),
            (-0.05392155159, 0.000000000, 37.54668956),
        ))*numericalunits.aBohr)
        
        testing.assert_equal(c.coordinates, numpy.array((
            ( 0.18921564, 0.25000000, 0.49621428),
            (-0.18921564,-0.25000000,-0.49621428),
            ( 0.43121489,-0.25000000,-0.42862875),
            (-0.43121492, 0.25000000, 0.42862876),
            (-0.07736811, 0.25000000,-0.40435873),
            ( 0.07736809,-0.25000000, 0.40435874),
        )))

        testing.assert_equal(c.values, ("W","W","Se","Se","Se","Se"))
       
    def test_valid_header(self):
        assert input.valid_header(self.parser.parser.string)
        
class Test_input1(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/elk.input.1.testcase"),'r') as f:
            self.parser = input(f.read())

    def test_unitCell(self):
        c = self.parser.unitCell()
        
        testing.assert_allclose(c.vectors, numpy.array((
            (5.944039, 0.      , 0.       ),
            (0.      , 3.301128, 0.       ),
            (0.      , 0.      , 20.160000),
        ))*numericalunits.aBohr*1.8897261245650618)
        
        testing.assert_equal(c.coordinates, numpy.array((
            (0.810843,  0.75,  0.495938),
            (0.189157,  0.25,  0.504062),
            (0.084550,  0.75,  0.593738),
            (0.577032,  0.25,  0.569474),
            (0.422968,  0.75,  0.430526),
            (0.915450,  0.25,  0.406262),
        )))

        testing.assert_equal(c.values, ("W","W","Se","Se","Se","Se"))

    def test_kp_path(self):
        kpp = self.parser.kp_path()
        
        testing.assert_equal(kpp.shape, (100,3))
        
        testing.assert_allclose(kpp[:,0], numpy.zeros(100))
        testing.assert_allclose(kpp[:,1], numpy.linspace(0,.5,100))
        testing.assert_allclose(kpp[:,2], numpy.zeros(100))

class Test_unitcells0(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/elk.unitcells.0.testcase"),'r') as f:
            self.parser = unitcells(f.read())
            
    def test_unitCells(self):
        c = self.parser.unitCells()
        assert len(c) == 13
        
        testing.assert_equal(c[0].vectors, numpy.array((
            (11.05491144, 0.000000000,-0.3046858492),
            (0.000000000, 6.194778086, 0.000000000),
            (-0.05392155159,0.000000000,37.54668956),
        ))*numericalunits.aBohr)
        
        testing.assert_equal(c[0].coordinates, numpy.array((
            ( 0.18925596, 0.25000000, 0.49618821),
            (-0.18925596,-0.25000000,-0.49618821),
            ( 0.43120320,-0.25000000,-0.42861786),
            (-0.43120323, 0.25000000, 0.42861787),
            (-0.07734583, 0.25000000,-0.40434314),
            ( 0.07734581,-0.25000000, 0.40434315),
        )))

        testing.assert_equal(c[0].values, ("W","W","Se","Se","Se","Se"))
        
        testing.assert_equal(c[-1].vectors, numpy.array((
            (11.39760697, 0.000000000,-0.491683336),
            (0.000000000, 6.657114335, 0.000000000),
            (0.2357896961,0.000000000, 36.87349199),
        ))*numericalunits.aBohr)
        
        testing.assert_equal(c[-1].coordinates, numpy.array((
            ( 0.18724733, 0.25000000, 0.49516462),
            (-0.18724733,-0.25000000,-0.49516462),
            ( 0.42544409,-0.25000000,-0.43179860),
            (-0.42544412, 0.25000000, 0.43179861),
            (-0.07848466, 0.25000000,-0.40573043),
            ( 0.07848464,-0.25000000, 0.40573044),
        )))

        testing.assert_equal(c[-1].values, ("W","W","Se","Se","Se","Se"))
       
    def test_valid_header(self):
        assert unitcells.valid_header(self.parser.parser.string[:1000])
        
class Test_output0(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/elk.output.0.testcase"),'r') as f:
            self.parser = output(f.read())

    def test_unitCell(self):
        c = self.parser.unitCell()
        
        testing.assert_equal(c.vectors, numpy.array((
            (11.39760697,  0.000000000, -0.4916833360),
            (0.000000000,  6.657114335,  0.000000000),
            (0.2407895939, 0.000000000,  36.87346002),
        ))*numericalunits.aBohr)
        
        testing.assert_equal(c.coordinates, numpy.array((
            ( 0.18724733,  0.25000000,  0.49516462),
            (-0.18724733, -0.25000000, -0.49516462),
            ( 0.42544409, -0.25000000, -0.43179860),
            (-0.42544412,  0.25000000,  0.43179861),
            (-0.07848466,  0.25000000, -0.40573043),
            ( 0.07848464, -0.25000000,  0.40573044),
        )))

        testing.assert_equal(c.values, ("W","W","Se","Se","Se","Se"))

    def test_reciprocal(self):
        r = self.parser.reciprocal()
        
        testing.assert_equal(r.vectors, numpy.array((
            (0.5511170734, 0.000000000, -0.003598882671),
            (0.000000000, 0.9438301629, 0.000000000),
            (0.007348783678, 0.000000000, 0.1703505934),
        ))*2*math.pi/numericalunits.aBohr)
       
    def test_valid_header(self):
        assert output.valid_header(self.parser.parser.string[:1000])
        
class Test_bands0(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/elk.bands.0.testcase"),'r') as f:
            self.parser = bands(f.read())

    def test_bands(self):
        b = self.parser.bands()
        
        testing.assert_equal(b.values.shape, (100,31714/101))

        testing.assert_equal(b.values[:3,0], numpy.array((-2.729662146, -2.729661982, -2.729661566))*2*numericalunits.Ry)
        testing.assert_equal(b.values[-3:,-1], numpy.array((0.8610216055, 0.8620398955, 0.8633321818))*2*numericalunits.Ry)
