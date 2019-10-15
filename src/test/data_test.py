import unittest
from dfttools.data import element_color_convention, element_number, element_for_number, element_size, element_mass

class DummyTest(unittest.TestCase):
    def test_defaults(self):
        print(element_color_convention[None])
        self.assertIsInstance(element_color_convention[None], tuple)
        self.assertEqual(len(element_color_convention[None]), 3)
        self.assertEqual(element_number[None], 0)
        self.assertEqual(element_for_number[-1], "??")
        self.assertIsInstance(element_size[None], tuple)
        self.assertEqual(len(element_size[None]), 2)
        self.assertIsInstance(element_mass[None], (int, float))
