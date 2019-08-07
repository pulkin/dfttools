import unittest
import pickle
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import numericalunits
import numpy
from numpy import testing

from dfttools.util import eval_nu, invert_nu, array, dumps, loads, dump, load, array_to_json


class EvalNUTest(unittest.TestCase):

    def test_nu(self):
        with self.assertRaises(ValueError):
            eval_nu("")
        with self.assertRaises(ValueError):
            eval_nu("nonexistent_unit")
        with self.assertRaises(ValueError):
            eval_nu("a+b")
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
        self.complex_sample = array([0, (1+.1j) / numericalunits.eV, 2 / numericalunits.eV], units="1/eV")
        self.none_sample = array([0, 1, 2])
        self.numpy_sample = numpy.array([1, 2, 3])
        self.all_samples = self.sample, self.complex_sample, self.none_sample

    def test_save_load(self):
        for i in range(len(self.all_samples)):
            data = pickle.dumps(self.all_samples[i])
            numericalunits.reset_units()
            x = pickle.loads(data)
            self.setUp()
            testing.assert_allclose(x, self.all_samples[i])

    def test_save_load_json(self):
        for i in range(len(self.all_samples)):
            s = StringIO()
            data = dumps(self.all_samples[i])
            dump(self.all_samples[i], s)
            s.seek(0)
            _data = s.read()
            assert data == _data
            s.seek(0)
            numericalunits.reset_units()
            x = loads(data)
            _x = load(s)
            testing.assert_equal(x, _x)
            self.setUp()
            testing.assert_allclose(x, self.all_samples[i])

    def test_serialization(self):
        for i in self.all_samples:
            serialized = i.to_json()
            test_data = i
            if i.units == "1/eV":
                test_data = i / (1. / numericalunits.eV)  # This expression is here due to round-off errors
            if numpy.iscomplexobj(test_data):
                test_data = numpy.vstack((test_data.real, test_data.imag)).T
            test_data = test_data.tolist()
            testing.assert_equal(serialized, dict(
                _type="numpy",
                data=test_data,
                complex=numpy.iscomplexobj(i),
                units=i.units,
            ))

    def test_serialization_numpy(self):
        serialized = array_to_json(self.numpy_sample)
        testing.assert_equal(serialized, dict(
            _type="numpy",
            data=self.numpy_sample.tolist(),
            complex=False,
            units=None,
        ))

    def test_serialization_fail(self):
        valid = dict(
            _type="numpy",
            data=[1, 2, 3],
            complex=False,
            units=None,
        )
        array.from_json(valid)
        with self.assertRaises(TypeError):
            array.from_json([valid])
        d = valid.copy()
        del d["_type"]
        with self.assertRaises(TypeError):
            array.from_json(d)
        with self.assertRaises(TypeError):
            array.from_json({**valid, **dict(_type="x")})

    def test_type(self):
        assert isinstance(self.sample.copy(), array)
        assert isinstance(self.sample / 2, array)
        assert isinstance(numpy.asanyarray(self.sample), array)
        assert isinstance(numpy.asanyarray(self.sample, dtype=int), array)
        assert isinstance(numpy.tile(self.sample, 2), array)
        mixed = self.sample + self.numpy_sample
        assert isinstance(mixed, array)
        assert mixed.units == "1/eV"
