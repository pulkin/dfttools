from itertools import chain

import unittest
from dfttools.data import element_color_convention, element_number, element_for_number, element_size, element_mass


class DummyTest(unittest.TestCase):
    def test_si(self):
        self.assertEqual(element_color_convention["si"], (240, 200, 160))
        self.assertEqual(element_number["si"], 14)
        self.assertEqual(element_for_number[14], "si")
        self.assertEqual(element_size["si"], (1.11, 1.11))
        self.assertEqual(element_mass["si"], 28.085)

    def test_color_convention(self):
        with self.assertRaises(RuntimeError):
            element_color_convention['x'] = 'y'

        for k in chain(element_color_convention, ("non-existing-item",)):
            v = element_color_convention[k]
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, tuple)
            self.assertEqual(len(v), 3)

    def test_element_number(self):
        with self.assertRaises(RuntimeError):
            element_number['x'] = 'y'

        for k in chain(element_number, ("non-existing-item",)):
            v = element_number[k]
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, int)

    def test_element_for_number(self):
        with self.assertRaises(RuntimeError):
            element_for_number['x'] = 'y'

        for k in chain(element_for_number, (-2,)):
            v = element_for_number[k]
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)

    def test_element_size(self):
        with self.assertRaises(RuntimeError):
            element_size['x'] = 'y'

        for k in chain(element_size, ("non-existing-item",)):
            v = element_size[k]
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, tuple)
            self.assertEqual(len(v), 2)

    def test_element_mass(self):
        with self.assertRaises(RuntimeError):
            element_mass['x'] = 'y'

        for k in chain(element_mass, ("non-existing-item",)):
            v = element_mass[k]
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, (float, int))
