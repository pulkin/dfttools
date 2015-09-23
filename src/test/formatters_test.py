import unittest

import numpy
from numpy import testing, random

from numericalunits import angstrom

from dfttools.types import Basis, UnitCell
from dfttools.parsers import structure, qe, openmx
from dfttools.formatters import xsf_structure, qe_input, siesta_input, openmx_input

class BackForthTests(unittest.TestCase):
    
    def setUp(self):
        self.cell = UnitCell(
            Basis((2.5*angstrom,2.5*angstrom,10*angstrom,0,0,.5), kind = 'triclinic'),
            (
                (1./3,1./3,.5),
                (2./3,2./3,.5),
            ),
            'C',
        )
        
    def test_xsf_back_forth(self):
        c1 = self.cell
        cells = structure.xsf(xsf_structure(c1)).unitCells()
        assert len(cells) == 1
        c2 = cells[0]
        assert c1.size() == c2.size()
        testing.assert_allclose(c1.vectors/angstrom, c2.vectors/angstrom, atol = 1e-6)
        testing.assert_allclose(c1.coordinates, c2.coordinates)
        testing.assert_equal(c1.values, c2.values)
        
    def test_xsf_back_forth_multi(self):
        c1 = []
        for i in range(10):
            c = self.cell.copy()
            c.coordinates += (numpy.random.rand(*c.coordinates.shape)-.5)/10
            c1.append(c)
        c2 = structure.xsf(xsf_structure(*c1)).unitCells()
        
        for i,j in zip(c1,c2):
            assert i.size() == j.size()
            testing.assert_allclose(i.vectors/angstrom, j.vectors/angstrom, atol = 1e-6)
            testing.assert_allclose(i.coordinates, j.coordinates)
            testing.assert_equal(i.values, j.values)

    def test_qe_back_forth(self):
        c1 = self.cell
        c2 = qe.input(qe_input(
            cell = c1,
            pseudopotentials = {"C":"C.UPF"},
        )).unitCell()
        assert c1.size() == c2.size()
        testing.assert_allclose(c1.vectors/angstrom, c2.vectors/angstrom, atol = 1e-6)
        testing.assert_allclose(c1.coordinates, c2.coordinates)
        testing.assert_equal(c1.values, c2.values)

    def test_siesta_not_raises(self):
        siesta_input(self.cell)

    def test_openmx_back_forth(self):
        c1 = self.cell
        c2 = openmx.input(openmx_input(
            c1,
            populations = {"C": "2 2"},
        )).unitCell()
        assert c1.size() == c2.size()
        testing.assert_allclose(c1.vectors/angstrom, c2.vectors/angstrom, atol = 1e-6)
        testing.assert_allclose(c1.coordinates, c2.coordinates, rtol = 1e-6)
        testing.assert_equal(c1.values, c2.values)

    def test_openmx_back_forth_negf_0(self):
        c1 = self.cell
        c2 = openmx.input(openmx_input(
            c1,
            l = c1,
            r = c1,
            populations = {"C": "2 2"},
        )).unitCell(l = c1, r = c1)
        assert c1.size() == c2.size()
        testing.assert_allclose(c1.vectors/angstrom, c2.vectors/angstrom, atol = 1e-6)
        testing.assert_allclose(c1.coordinates, c2.coordinates, rtol = 1e-6)
        testing.assert_equal(c1.values, c2.values)
        
    def test_openmx_back_forth_negf_1(self):
        c1 = self.cell.repeated(2,1,1)
        l = self.cell
        r = self.cell.repeated(3,1,1)
        c2 = openmx.input(openmx_input(
            c1,
            l = l,
            r = r,
            populations = {"C": "2 2"},
        )).unitCell(l = l, r = r)
        assert c1.size() == c2.size()
        testing.assert_allclose(c1.vectors/angstrom, c2.vectors/angstrom, atol = 1e-6)
        testing.assert_allclose(c1.coordinates, c2.coordinates, rtol = 1e-6)
        testing.assert_equal(c1.values, c2.values)
