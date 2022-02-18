import os
import unittest

from dfttools.parsers import qe, openmx, elk, structure, tools, wannier90
from dfttools.parsers.generic import IdentifiableParser, ParseError
from dfttools.simple import get_all_parsers, guess_parser, parse


class Test_methods(unittest.TestCase):

    def setUp(self):
        self.data = (
            ("elk.input.0.testcase", elk.Input),
            ("elk.input.1.testcase", elk.Input),
            ("elk.output.0.testcase", elk.Output),
            ("elk.unitcells.0.testcase", elk.CellsParser),
            ("openmx.input.0.testcase", openmx.Input),
            ("openmx.input.1.testcase", openmx.Input),
            ("openmx.input.2.lead.testcase", openmx.Input),
            ("openmx.input.2.scatter.testcase", openmx.Input),
            ("openmx.output.0.testcase/output", openmx.Output),
            ("openmx.tran.0.testcase", openmx.Transmission),
            ("openmx.tran.1.testcase", openmx.Transmission),
            ("openmx.md.0.testcase", openmx.MD),
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
            ("dfttools.0.testcase", tools.JSONStorage),
            ("structure.cif.0.testcase", structure.CIF),
            ("wannier.input.0.testcase", wannier90.input),
        )

    def test_get_all_parsers(self):
        for i in get_all_parsers():
            assert issubclass(i, IdentifiableParser)

    def test_guess_parser(self):
        for f, p in self.data:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "parsers/cases/" + f)
            with open(path, "r") as fl:
                parsers = guess_parser(fl)
                self.assertEqual(tuple(parsers), (p,))

    def test_parse_0(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "parsers/cases/qe.output.0.testcase")
        with open(path, 'r') as f:
            c1 = qe.output(f.read()).cells()
            c2 = parse(f, "unit-cell")

        for i, j in zip(c1, c2):
            assert i == j

    def test_parse_1(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "parsers/cases/openmx.output.0.testcase/output")
        p2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "parsers/cases/openmx.output.0.testcase/input")
        with open(p2, 'r') as f:
            c = parse(f, "unit-cell")
        with open(path, 'r') as f:
            c1 = openmx.output(f.read()).cells(c)
            c2 = parse(f, "unit-cell")

        for i, j in zip(c1, c2):
            assert i == j


class TestNameGuess(unittest.TestCase):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "parsers/cases/openmx.bands.0.testcase")

    def setUp(self):
        with open(self.path, 'r') as f:
            with open(self.path + '.Band', 'w') as f2:
                f2.write(f.read())

    def tearDown(self):
        os.remove(self.path + '.Band')

    def test_guess_by_name(self):
        with open(self.path + '.Band', 'r') as f:
            parsers = guess_parser(f)
            assert len(parsers) == 1
            assert parsers[0] == openmx.Bands


class TestParserFailures(unittest.TestCase):
    file_no_band_structure = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                          "parsers/cases/elk.input.1.testcase")
    __file_truncated_structure__ = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                "parsers/cases/structure.xsf.0.testcase")
    file_truncated_structure = __file_truncated_structure__ + "_TEMP"

    def setUp(self):
        with open(self.__file_truncated_structure__, 'r') as f:
            with open(self.file_truncated_structure, 'w') as f2:
                f2.write(f.read(128))

    def tearDown(self):
        os.remove(self.file_truncated_structure)

    def test_parse_failed_0(self):
        with open(self.file_no_band_structure, 'r') as f:
            with self.assertRaises(ParseError):
                parse(f, 'band-structure')
            with self.assertRaises(ParseError):
                parse(f, 'some-non-existing-data')

    def test_parse_failed_1(self):
        with open(self.__file_truncated_structure__, 'r') as f:
            parse(f, 'unit-cell')
        with open(self.file_truncated_structure, 'r') as f:
            with self.assertRaises(ParseError):
                parse(f, 'unit-cell')
