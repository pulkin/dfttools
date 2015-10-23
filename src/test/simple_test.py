import os

import unittest

from dfttools.simple import get_all_parsers, guess_parser, parse
from dfttools.parsers.generic import AbstractParser
from dfttools.parsers import qe, openmx, elk, structure

class Test_methods(unittest.TestCase):
    
    def setUp(self):
        self.data = (
            ("elk.input.0.testcase", elk.Input),
            ("elk.input.1.testcase", elk.Input),
            ("elk.output.0.testcase", elk.Output),
            ("elk.unitcells.0.testcase", elk.UnitCellsParser),
            ("openmx.input.0.testcase", openmx.Input),
            ("openmx.input.1.testcase", openmx.Input),
            ("openmx.input.2.lead.testcase", openmx.Input),
            ("openmx.input.2.scatter.testcase", openmx.Input),
            ("openmx.output.0.testcase/output", openmx.Output),
            ("openmx.tran.0.testcase", openmx.Transmission),
            ("openmx.tran.1.testcase", openmx.Transmission),
            ("qe.bands.0.testcase", qe.Bands),
            ("qe.cond.0.testcase", qe.Cond),
            ("qe.cond.1.testcase", qe.Cond),
            ("qe.cond.2.testcase", qe.Cond),
            ("qe.cond.3.testcase", qe.Cond),
            ("qe.cond.4.testcase", qe.Cond),
            ("qe.output.0.testcase", qe.Output),
            ("qe.output.1.testcase", qe.Output),
            ("qe.output.2.testcase", qe.Output),
            ("qe.output.3.testcase", qe.Output),
            ("qe.output.4.testcase", qe.Output),
            ("qe.proj.0.testcase", qe.Proj),
            ("qe.proj.1.testcase", qe.Proj),
            ("structure.xsf.0.testcase", structure.XSF),
            ("structure.xsf.1.testcase", structure.XSF),
            ("structure.xsf.2.testcase", structure.XSF),
        )

    def test_get_all_parsers(self):
        for i in get_all_parsers():
            assert issubclass(i, AbstractParser)

    def test_guess_parser(self):
        for f, p in self.data:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"parsers/cases/"+f)
            with open(path, "r") as fl:
                parsers = guess_parser(fl)
                assert len(parsers) == 1
                assert parsers[0] == p
        
    def test_parse_0(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"parsers/cases/qe.output.0.testcase")
        with open(path,'r') as f:
            c1 = qe.output(f.read()).unitCells()
            c2 = parse(f,"unit-cell")
        
        for i,j in zip(c1,c2):
            assert i==j

    def test_parse_1(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"parsers/cases/openmx.output.0.testcase/output")
        p2 = os.path.join(os.path.dirname(os.path.realpath(__file__)),"parsers/cases/openmx.output.0.testcase/input")
        with open(p2, 'r') as f:
            c = parse(f, "unit-cell")
        with open(path,'r') as f:
            c1 = openmx.output(f.read()).unitCells(c)
            c2 = parse(f,"unit-cell")
        
        for i,j in zip(c1,c2):
            assert i==j
