import unittest
import pickle

import numericalunits
import numpy
from numpy import testing

from dfttools.util import eval_nu, invert_nu, array, dumps, loads


class EvalNUTest(unittest.TestCase):

    def test_nu(self):
        with self.assertRaises(ValueError):
            eval_nu("")
        with self.assertRaises(ValueError):
            eval_nu("nonexistent_unit")
        testing.assert_equal(eval_nu("angstrom"), numericalunits.angstrom)
        eva = numericalunits.eV / numericalunits.angstrom
        for i in ("eV/angstrom", "eV /angstrom", "eV/ angstrom", "eV / angstrom", "eV/angstrom ", " eV/angstrom",
                  " eV  /   angstrom  "):
            testing.assert_equal(eval_nu(i), eva)

        with self.assertRaises(ValueError):
            eval_nu("eV/nonexistent_unit")
        testing.assert_equal(eval_nu("1/angstrom"), 1./numericalunits.angstrom)

    def test_invert(self):
        testing.assert_equal(invert_nu("angstrom"), "1/angstrom")
        testing.assert_equal(invert_nu("1/angstrom"), "angstrom")
        testing.assert_equal(invert_nu("angstrom/eV"), "1/angstrom*eV")
        testing.assert_equal(invert_nu("angstrom/eV/Hartree"), "1/angstrom*eV*Hartree")
        testing.assert_equal(invert_nu("1/1/angstrom/eV/Hartree"), "angstrom*eV*Hartree")


class ArrayUnitsTest(unittest.TestCase):

    def setUp(self):
        self.sample = array([0, 1. / numericalunits.eV, 2. / numericalunits.eV], units="1/eV")
        self.complex_sample = array([0, 1+.1j, 2], units="1/eV")

    def test_save_load(self):
        data = pickle.dumps(self.sample)
        numericalunits.reset_units()
        x = pickle.loads(data)
        self.setUp()
        testing.assert_allclose(x, self.sample)

    def test_save_load_json(self):
        data = dumps(self.sample)
        numericalunits.reset_units()
        x = loads(data)
        self.setUp()
        testing.assert_allclose(x, self.sample)

    def test_serialization(self):
        serialized = self.sample.to_json()
        testing.assert_equal(serialized, dict(
            _type="numpy",
            data=(self.sample / (1. / numericalunits.eV)).tolist(),
            complex=False,
            units="1/eV",
        ))

    def test_serialization_complex(self):
        serialized = self.complex_sample.to_json()
        tmp = self.complex_sample / (1. / numericalunits.eV)
        tmp = numpy.concatenate((tmp.real[..., numpy.newaxis], tmp.imag[..., numpy.newaxis]), axis=-1)
        testing.assert_equal(serialized, dict(
            _type="numpy",
            data=tmp.tolist(),
            complex=True,
            units="1/eV",
        ))

    def test_type(self):
        assert isinstance(self.sample.copy(), array)
        assert isinstance(self.sample / 2, array)
        assert isinstance(numpy.asanyarray(self.sample), array)
        assert isinstance(numpy.asanyarray(self.sample, dtype=int), array)
        assert isinstance(numpy.tile(self.sample, 2), array)
