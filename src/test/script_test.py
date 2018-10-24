import os
import subprocess
import sys
import unittest


def get_fname(self, id):
    return "__" + self.__class__.__name__ + "TEMP" + str(id)


class Test_dft_plot_bands(unittest.TestCase):

    def setUp(self):
        fname = get_fname(self, "0.pdf")
        if os.path.exists(fname):
            os.remove(fname)

    tearDown = setUp

    def test_output(self):
        fname = get_fname(self, "0.pdf")
        assert subprocess.call((
            sys.executable,
            "scripts/dft-plot-bands",
            "test/parsers/cases/qe.output.0.testcase",
            "-o",
            fname,
        )) == 0
        assert os.path.exists(fname)


class Test_dft_svg_struct(unittest.TestCase):

    def setUp(self):
        fname = get_fname(self, "0.svg")
        if os.path.exists(fname):
            os.remove(fname)

    tearDown = setUp

    def test_output(self):
        fname = get_fname(self, "0.svg")
        assert subprocess.call((
            sys.executable,
            "scripts/dft-svg-structure",
            "test/parsers/cases/qe.output.0.testcase",
            fname,
        )) == 0
        assert os.path.exists(fname)
