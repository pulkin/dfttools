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
        this = os.path.dirname(os.path.realpath(__file__))
        env = os.environ.copy()
        env["PYTHONPATH"] = "{}:{}".format(
            env["PYTHONPATH"],
            os.path.join(this, "../"),
        )
        p = subprocess.Popen((
            sys.executable,
            os.path.join(this, "../scripts/dft-plot-bands"),
            os.path.join(this, "parsers/cases/qe.output.0.testcase"),
            "-o",
            fname,
        ), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        output, error = p.communicate("")
        if len(output) > 0:
            print(output)
        if len(error) > 0:
            print(error)
        assert p.returncode == 0
        assert os.path.exists(fname)


class Test_dft_svg_struct(unittest.TestCase):

    def setUp(self):
        fname = get_fname(self, "0.svg")
        if os.path.exists(fname):
            os.remove(fname)

    tearDown = setUp

    def test_output(self):
        fname = get_fname(self, "0.svg")
        this = os.path.dirname(os.path.realpath(__file__))
        env = os.environ.copy()
        env["PYTHONPATH"] = "{}:{}".format(
            env["PYTHONPATH"],
            os.path.join(this, "../"),
        )
        p = subprocess.Popen((
            sys.executable,
            os.path.join(this, "../scripts/dft-svg-structure"),
            os.path.join(this, "parsers/cases/qe.output.0.testcase"),
            fname,
        ), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        output, error = p.communicate("")
        if len(output) > 0:
            print(output)
        if len(error) > 0:
            print(error)
        assert p.returncode == 0
        assert os.path.exists(fname)
