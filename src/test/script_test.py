import unittest
import os, sys, subprocess

class Test_dft_plot_bands(unittest.TestCase):

    def get_fname(self, id):
        return "__"+self.__class__.__name__+"TEMP"+str(id)
        
    def setUp(self):
        fname = self.get_fname("0.pdf")
        if os.path.exists(fname):
            os.remove(fname)
            
    tearDown = setUp
        
    def test_output(self):
        fname = self.get_fname("0.pdf")
        assert subprocess.call((
            sys.executable,
            "scripts/dft-plot-bands",
            "test/parsers/cases/qe.output.0.testcase",
            "-o",
            fname,
        )) == 0
        assert os.path.exists(fname)
