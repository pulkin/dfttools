import unittest
import os
import math

from numpy import testing
import numpy
import numericalunits

from dfttools.parsers.openmx import bands, output, input, transmission, populations, joint_populations, JSON_DOS, Transmission
from dfttools.types import UnitCell, Basis

class Test_bands0(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/openmx.bands.0.testcase"),'r') as f:
            self.parser = bands(f.read())

    def test_fermi(self):
        assert self.parser.fermi() == -0.161133*2*numericalunits.Ry
        
    def test_captions(self):
        testing.assert_equal(self.parser.captions(), {
            0:"M",
            99:"G",
            100:"G",
            199:"K",
            200:"K",
            249:"M",
        })
        
    def test_bands(self):
        b = self.parser.bands()
        testing.assert_equal(b.vectors,numpy.array((
            (1.011228, -0.583833, 0.000000),
            (0.000000, 1.167666, 0.000000),
            (0.000000, 0.000000, 0.033249),
        ))*2*math.pi/numericalunits.aBohr)
        assert b.coordinates.shape[0] == 250
        assert b.values.shape[1] == 118
        
        testing.assert_equal(b.coordinates[0,:],(0.500001, -0.000001, 0.000002))
        testing.assert_allclose(b.coordinates[0,:], b.coordinates[-1,:])
        
        testing.assert_allclose(b.values[0,:],(numpy.array((-2.424179439819485,
            -2.424179439795330, -1.537117009662608, -1.537117006340471,
            -1.449284539430810, -1.449284537415585, -1.444805325728458,
            -1.444805323647778, -0.647041898384984, -0.647041892412120,
            -0.641956689127265, -0.641956688537892, -0.398372289761924,
            -0.398372284588964, -0.368313841617996, -0.368313823726870,
            -0.331063155223418, -0.331063154791907, -0.291596980253905,
            -0.291596964923972, -0.263885982743258, -0.263885963251231,
            -0.217730274903104, -0.217730040506557, -0.212005955051399,
            -0.212005689294971, -0.119530208759314, -0.119530195802536,
            -0.104171661177456, -0.104171615323529, -0.060173882829107,
            -0.060173864363899, -0.044110822885303, -0.044110819596738,
             0.023676269044778,  0.023676281361570,  0.088727552579903,
             0.088727553156531,  0.109027679665700,  0.109027691291708,
             0.169189713437080,  0.169189715448858,  0.183435557043469,
             0.183435579143966,  0.224028762657754,  0.224028771770464,
             0.400498726817328,  0.400498728918152,  0.434864763826354,
             0.434864769855039,  0.462844460651656,  0.462844477476227,
             0.514672883961469,  0.514672915030914,  0.584606953295579,
             0.584606962631887,  0.627230160559928,  0.627230199539481,
             0.629033323773423,  0.629033340909780,  0.650878551167126,
             0.650878570165002,  0.680062373390613,  0.680062375730062,
             0.728021193078287,  0.728021226780865,  0.739338173837054,
             0.739338202736014,  0.739484521863883,  0.739484544152468,
             0.810127522240378,  0.810127538477089,  0.823951657619717,
             0.823951659007691,  0.936303051089771,  0.936303086173654,
             0.955098053379815,  0.955098059710558,  1.007258340331402,
             1.007258343510678,  1.015473262113421,  1.015473262194476,
             1.026714225067395,  1.026714225580743,  1.084678912528088,
             1.084678915535665,  1.133460643653227,  1.133460645657584,
             1.143860304940359,  1.143860310211632,  1.166541914868326,
             1.166541915543972,  1.186521946136983,  1.186521947918524,
             1.216065632113939,  1.216065633237035,  1.269310183902559,
             1.269310186467420,  1.336105002473844,  1.336105004401566,
             1.344736163245268,  1.344736163394513,  1.351050988628181,
             1.351050993825189,  1.449167823561218,  1.449167846786868,
             1.462964038659094,  1.462964040710026,  1.557730250666156,
             1.557730277121358,  2.148239640563523,  2.148239843439056,
             2.178107733188863,  2.178107969469329,  2.310569568913024,
             2.310569586463968,  2.675051299060441,  2.675051340334891)))*numericalunits.Ry*2
        )
        testing.assert_allclose(b.values[0,:],b.values[-1,:])
        assert b.meta["Fermi"] == -0.161133*numericalunits.Ry*2
        assert b.meta["special-points"] == {
            0:"M",
            99:"G",
            100:"G",
            199:"K",
            200:"K",
            249:"M",
        }
        
class Test_input0(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/openmx.input.0.testcase"),'r') as f:
            data = f.read()
            self.parser = input(data)
            self.parser_au = input(data.replace("Ang","au").replace("Frac","au"))
       
    def test_valid_header(self):
        assert input.valid_header(self.parser.parser.string[:1000])

    def test_getWord(self):
        assert self.parser.getWord("Atoms.SpeciesAndCoordinates.Unit") == "Frac"
        
    def test_getNonSpaced(self):
        assert self.parser.getNonSpaced("system.name") == "mose2_1l"
        
    def test_systemName(self):
        assert self.parser.systemName() == "mose2_1l"
        
    def test_getFloat(self):
        assert self.parser.getFloat("scf.energycutoff") == 200.0
        
    def test_getInt(self):
        assert self.parser.getInt("scf.maxIter") == 100
        
    def test_unitCell(self):
        c = self.parser.unitCell()
        
        testing.assert_equal(c.vectors,numpy.array((
            (3.288, 0.0, 0.0),
            (1.644, 2.847491528, 0.0),
            (0.0, 0.0, 100.0),
        ))*numericalunits.angstrom)
        
        assert c.coordinates.shape[0] == 3
        
        testing.assert_equal(c.coordinates,numpy.array((
            (0.33333333333353,    0.33333333333331,    0.50000000000001),
            (0.66666666666614,    0.66666666666674,    0.48313110391249),
            (0.66666666666614,    0.66666666666674,    0.51686889613927),
        )))
        
        assert c.values[0] == "Mo"
        assert c.values[1] == "Se"
        assert c.values[2] == "Se"
        
    def test_unitCell_au(self):
        c = self.parser_au.unitCell()
        
        testing.assert_equal(c.vectors,numpy.array((
            (3.288, 0.0, 0.0),
            (1.644, 2.847491528, 0.0),
            (0.0, 0.0, 100.0),
        ))*numericalunits.aBohr)

        testing.assert_allclose(c.cartesian(),numpy.array((
            (0.33333333333353,    0.33333333333331,    0.50000000000001),
            (0.66666666666614,    0.66666666666674,    0.48313110391249),
            (0.66666666666614,    0.66666666666674,    0.51686889613927),
        ))*numericalunits.aBohr)

class Test_input1(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/openmx.input.1.testcase"),'r') as f:
            self.parser = input(f.read())

    def test_unitCell(self):
        c = self.parser.unitCell()
        
        testing.assert_equal(c.vectors,numpy.array((
            (55.66990773701619, 0.0, 0.0),
            (0.0, 3.32493478114427, 0.0),
            (0.0, 0.0, 100.0),
        ))*numericalunits.angstrom)
        
        assert c.coordinates.shape[0] == 58
        
        testing.assert_allclose(c.cartesian()[0],numpy.array((0.95982599546579, 0.83123369528605, 50.0))*numericalunits.angstrom)
        testing.assert_allclose(c.cartesian()[-1],numpy.array((54.71008174155039, 0.83123369528605, 50.0))*numericalunits.angstrom)
        
        assert c.values[0] == "mo"
        assert c.values[1] == "se"
        assert c.values[2] == "se"
        
        assert c.values[-3] == "se"
        assert c.values[-2] == "se"
        assert c.values[-1] == "mo"
        
class Test_input2(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/openmx.input.2.scatter.testcase"),'r') as f:
            data = f.read()
            self.s = input(data)
            self.s_au = input(data.replace("Ang","AU"))
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/openmx.input.2.lead.testcase"),'r') as f:
            data = f.read()
            self.l = input(data)
            self.l_au = input(data.replace("Ang","AU"))
            
    def test_unitCell(self):
        with self.assertRaises(ValueError):
            self.s.unitCell()
        
        l = self.l.unitCell()
        s = self.s.unitCell(l = l, r = l)
        
        testing.assert_allclose(l.vectors, s.vectors)
        testing.assert_allclose(l.coordinates, s.coordinates)
        self.assertSequenceEqual(tuple(l.values),tuple(s.values))
        
    def test_unitCell_au(self):
        with self.assertRaises(ValueError):
            self.s_au.unitCell()
        
        l = self.l_au.unitCell()
        s = self.s_au.unitCell(l = l, r = l)
        
        testing.assert_allclose(l.vectors, s.vectors)
        testing.assert_allclose(l.coordinates, s.coordinates)
        self.assertSequenceEqual(tuple(l.values),tuple(s.values))
        
    def test_tolerance(self):
        l = self.l.unitCell()
        l.coordinates[2,2] += 1e-6
        with self.assertRaises(ValueError):
            self.s.unitCell(l = l, r = l)
            
        self.s.unitCell(l = l, r = l, tolerance = 0.01)
        
class Test_output_invalid(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/qe.output.0.testcase"),'r') as f:
            self.parser = output(f.read())
            
    def test_invalid_header(self):
        assert not output.valid_header(self.parser.parser.string[:1000])

class Test_output0(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/openmx.output.0.testcase/output"),'r') as f:
            self.parser = output(f.read())
       
    def test_valid_header(self):
        assert output.valid_header(self.parser.parser.string[:1000])
        
    def test_version(self):
        assert self.parser.version() == "3.7.8"
        
    def test_total(self):
        e = self.parser.total()
        
        testing.assert_allclose(e, numpy.array((
            -89.989525394506, -89.991070546406, -89.992311708479,
            -89.993300200069, -89.994078756978, -89.994726796551,
            -89.996202786290, -89.996789948618, -89.996801412745,
            -89.996799134762, -89.996801452498, -89.996801529672,
            -89.996801486036, -89.996801501449,
        ))*2*numericalunits.Ry)
        
    def test_unitCells(self):
        vecs = numpy.array((
            (3.7545085, 0.0, 0.0),
            (1.87725425, 3.25149973972460681892, 0.0),
            (0.0, 0.0, 100)
        ))*numericalunits.angstrom
        
        c = self.parser.unitCells(UnitCell(
            Basis(vecs),
            (
                (0.33333333333353, 0.33333333333331, 0.50000000000001),
                (0.66666666666614, 0.66666666666674, 0.48313110391249),
                (0.66666666666614, 0.66666666666674, 0.51686889613927),
            ),
            ("mo","se","se"),
        ))
        
        assert len(c) == 14
        
        for cc in c:
            assert cc.coordinates.shape[0] == 3
            
            testing.assert_allclose(cc.vectors,vecs)
            
            assert cc.values[0] == "mo"
            assert cc.values[1] == "se"
            assert cc.values[2] == "se"
        
        testing.assert_allclose(c[0].cartesian(), numpy.array((
            (1.8773, 1.0838, 50.0000),
            (3.7545, 2.1677, 48.3278),
            (3.7545, 2.1677, 51.6722),
        ))*numericalunits.angstrom)
        
        testing.assert_allclose(c[-1].cartesian(), numpy.array((
            (1.8773, 1.0838, 50.0000),
            (3.7545, 2.1677, 48.4420),
            (3.7545, 2.1677, 51.5580),
        ))*numericalunits.angstrom)
        
    def test_populations(self):
        p = self.parser.populations()
        
        self.assertSequenceEqual(p.shape,(301,3))
        testing.assert_allclose(p[0],numpy.array((12.13,6.93,6.93))*26/25.99)
        testing.assert_allclose(p[-1],(13.6,6.2,6.2))

    def test_solverd(self):
        s = self.parser.solvers()
        
        for i in range(301):
            assert s[i] == "Band_DFT"
        
    def test_convergence(self):
        c = self.parser.convergence()
        
        self.assertSequenceEqual(c.shape,(301,))
        assert c[0] == 1.0
        assert c[-1] == 2.7e-11

class Test_transmission_filename(unittest.TestCase):
    
    def test(self):
        assert Transmission.valid_filename("abc.tran3_5")
        assert Transmission.valid_filename("abc.tran0_0")
        assert Transmission.valid_filename(".tran.tran.tran0_0")
        assert not Transmission.valid_filename("abc.tran-1_0")
        assert not Transmission.valid_filename("abc.tran10")
        assert not Transmission.valid_filename("abc.tran10_")
        assert not Transmission.valid_filename("abc.tran_10")

class Test_transmission_scalar(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/openmx.tran.0.testcase"),'r') as f:
            self.parser = transmission(f.read())
       
    def test_valid_header(self):
        assert transmission.valid_header(self.parser.parser.string[:1000])

    def test_total(self):
        total = self.parser.total()
        testing.assert_allclose(total, numpy.array((1.801950e-01, 1.825443e-01,
            1.837573e-01, 1.832859e-01, 1.810185e-01, 1.771737e-01,
            1.719866e-01, 1.652656e-01, 1.559525e-01, 1.419830e-01,
            1.210971e-01, 9.319536e-02, 6.267551e-02, 3.655321e-02,
            1.893359e-02, 9.081443e-03, 4.199289e-03, 1.926708e-03,
            8.921274e-04, 4.202717e-04, 2.019035e-04, 9.877827e-05,
            4.900993e-05, 2.449388e-05, 1.220245e-05, 5.957481e-06,
            2.761137e-06, 1.126768e-06, 3.023642e-07, 3.006437e-20))*2)
            
    def test_energy(self):
        energy = self.parser.energy()
        testing.assert_allclose(energy, numpy.linspace(-1.05, -0.7, 30)*numericalunits.eV, rtol=1e-6)
        
class Test_transmission_noncoll(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/openmx.tran.1.testcase"),'r') as f:
            self.parser = transmission(f.read())
       
    def test_valid_header(self):
        assert transmission.valid_header(self.parser.parser.string[:1000])

    def test_total(self):
        total = self.parser.total()
        testing.assert_allclose(total, (0.,)*5+(1,2,2,3,4), atol = 1e-4)
            
    def test_energy(self):
        energy = self.parser.energy()
        testing.assert_allclose(energy, numpy.linspace(-0.8, -1.6, 10)*numericalunits.eV, rtol=1e-6)
        
class Test_lowdin(unittest.TestCase):
    
    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/openmx.lowdin.0.testcase"),'r') as f:
            self.data = populations(f.read())
            
    def test_consistency(self):
        shape = self.data["weights"].shape
        testing.assert_equal(self.data["bands"].shape, (shape[0],))
        testing.assert_equal(self.data["energies"].shape, (shape[0],))
        for v in self.data["basis"].values():
            testing.assert_equal(v.shape, (shape[1],))

class Test_joint_lowdin(unittest.TestCase):
    
    def setUp(self):
        self.data = []
        for i in range(4):
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/openmx.lowdin.{:d}.testcase".format(i)),'r') as f:
                self.data.append(f.read())
            
    def test_consistency(self):
        data = joint_populations(self.data[1:])
        shape = data["weights"].shape
        assert shape[0] == 3
        testing.assert_equal(data["bands"].shape, (shape[1],))
        testing.assert_equal(data["energies"].shape, (shape[0],shape[1]))
        for v in data["basis"].values():
            testing.assert_equal(v.shape, (shape[2],))
            
    def test_fail_0(self):
        with self.assertRaises(ValueError):
            joint_populations([self.data[0],self.data[1]])
            
    def test_fail_1(self):
        with self.assertRaises(ValueError):
            joint_populations([self.data[1],self.data[3]])
            
class Test_JSON_DOS(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cases/openmx.dos.0.testcase"),'r') as f:
            self.parser = JSON_DOS(f.read())
            
    def test_consistency(self):
        shape = self.parser.weights().shape
        
        testing.assert_equal(shape[:3], [50,1,50])
        
        for v in self.parser.basis().values():
            testing.assert_equal(v.shape, (shape[-1],))

        testing.assert_allclose(self.parser.energies(),numpy.linspace(-numericalunits.eV, numericalunits.eV, 50), rtol = 1e-5)
        testing.assert_equal(self.parser.ky(),numpy.linspace(-0.49, 0.49, 50))
        testing.assert_equal(self.parser.kz(),[0])
        
