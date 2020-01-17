import math
import re
import unittest

from dfttools.parsers.generic import parse, cre_var_name, cre_word
from numpy import testing


class StringTest(unittest.TestCase):

    def test_goto_0(self):
        sp = parse("This is a big string with multiple occurences of this: \"this, this and thIs\".")
        assert sp.__position__ == 0
        sp.goto("is")
        assert sp.__position__ == 2
        sp.goto("is a")
        assert sp.__position__ == 5
        sp.goto("this")
        sp.__position__ += 1
        sp.goto("this")
        sp.__position__ += 1
        sp.goto("this")
        sp.__position__ += 1
        sp.goto("this")
        assert sp.__position__ == len(sp.string) - 6
        sp.__position__ += 1
        self.assertRaises(StopIteration, sp.goto, "this")

    def test_goto_1(self):
        sp = parse("ABC, Abc, abc, aBc, aBC")
        sp.goto("abC")
        assert sp.__position__ == 0
        sp.goto("aBc")
        assert sp.__position__ == 0
        sp.goto(re.compile("aBc"))
        assert sp.__position__ == 15
        self.assertRaises(StopIteration, sp.goto, "bac")

    def test_pop_save(self):
        sp = parse("a, b, c, d, e, f")
        self.assertRaises(Exception, sp.pop)
        sp.save()
        sp.goto("b")
        sp.save()
        sp.goto("c")
        assert sp.__position__ == 6
        sp.pop()
        assert sp.__position__ == 3
        sp.pop()
        assert sp.__position__ == 0
        self.assertRaises(Exception, sp.pop)
        sp.fw(12)
        assert sp.__position__ == 12
        sp.fw(4)
        assert  sp.__position__ == 16
        self.assertRaises(Exception, sp.fw, 1)

    def test_skip(self):
        sp = parse("ABC, Abc, abc, aBc, aBC, aBC, aBC")
        sp.skip("abc")
        assert sp.__position__ == 3
        sp.skip("abc", n=2)
        assert sp.__position__ == 13
        sp.skip(re.compile("aBC"), n=2)
        assert sp.__position__ == 28
        self.assertRaises(StopIteration, sp.skip, re.compile("Abc"))

    def test_skipAll(self):
        sp = parse("ABC, ABC, abc, aBc, aBC, aBC, aBC, def")
        self.assertRaises(StopIteration, sp.skip_all, "xyz")
        sp.skip_all(re.compile("ABC"))
        assert sp.__position__ == 8
        sp.skip_all("abc")
        assert sp.__position__ == 33
        self.assertRaises(StopIteration, sp.skip_all, "abc")

    def test_present(self):
        sp = parse("abc,def,ghi")
        sp.__position__ = 3
        assert sp.present("def")
        assert sp.present("def,")
        assert sp.present("ghi")
        assert not sp.present("weird")
        assert not sp.present("abc")

    def test_distance(self):
        sp = parse("abc,def,ghi,abc")
        assert sp.distance("abc") == 0
        assert sp.distance("abc", n=2) == 12
        self.assertRaises(StopIteration, sp.distance, "abc", n=3)
        assert sp.distance("abc", n=4, default=-1) == -1
        sp.__position__ = 1
        assert sp.distance("abc") == 11
        assert sp.distance("abc", default=-1) == 11

    def test_nextInt(self):
        sp = parse("123abc456 78.8 +45.68-73 1 2 3 4 5 6 7 8 abc 9")
        assert sp.next_int() == 123
        assert sp.next_int() == 456
        assert sp.next_int() == 78
        assert sp.next_int() == 8
        testing.assert_equal(sp.next_int(3), (45, 68, -73))
        testing.assert_equal(sp.next_int((2, 3)), ((1, 2, 3), (4, 5, 6)))
        testing.assert_equal(sp.next_int("abc"), (7, 8))
        assert sp.next_int() == 9
        self.assertRaises(StopIteration, sp.next_int)

    def test_nextFloat(self):
        sp = parse("123, 12a3, nan, 5, nan, 4.66, 12., 356.2, 4e-2, 4.e-5, 56.2E-2 -36.6 1.0 2.0 3.0 abc 4.0")
        assert sp.next_float() == 123
        testing.assert_equal(sp.next_float(2), (12, 3))
        assert math.isnan(sp.next_float())
        testing.assert_equal(sp.next_float(2), (5, float("nan")))
        testing.assert_equal(sp.next_float((2, 3)), ((4.66, 12., 356.2), (4e-2, 4e-5, 56.2e-2)))
        assert sp.next_float() == -36.6
        testing.assert_equal(sp.next_float("abc"), (1., 2., 3.))
        assert sp.next_float() == 4.0
        self.assertRaises(StopIteration, sp.next_float)

    def test_nextFloat2(self):
        sp = parse("123, 456; 789")
        testing.assert_equal(sp.next_float(";"), (123, 456))

    def test_lineOperations(self):
        sp = parse("\none\ntwo\nthree\nfour")
        sp.rtn()
        assert sp.__position__ == 0
        assert sp.next_line() == ""
        assert sp.__position__ == 1
        sp.skip("two")
        sp.rtn()
        assert sp.__position__ == 5
        testing.assert_equal(sp.next_line(2), ("two", "three"))
        assert sp.__position__ == 15
        sp.rtn()
        assert sp.__position__ == 15
        sp.skip("four")
        sp.rtn()
        assert sp.__position__ == 15
        assert sp.next_line() == "four"
        self.assertRaises(StopIteration, sp.next_line)

    def test_lineOperations2(self):
        sp = parse("Hello\n World!")
        sp.__position__ = 2
        sp.rtn()
        assert sp.__position__ == 0

        sp.__position__ = 10
        sp.rtn()
        assert sp.__position__ == 6

    def test_nextMatch(self):
        sp = parse("one two2 three, four\n  five   six seven")
        assert sp.next_match(cre_word) == "one"
        assert sp.__position__ == 3
        assert sp.next_match(cre_word) == "two2"
        assert sp.__position__ == 8
        assert sp.next_match(cre_word) == "three"
        assert sp.next_match(cre_word) == "four"
        assert sp.next_match(cre_word) == "five"
        assert sp.next_match(cre_word) == "six"
        assert sp.next_match(cre_word) == "seven"
        self.assertRaises(StopIteration, sp.next_match, cre_word)

    def test_nextMatch2(self):
        sp = parse("one two2 three, four\n  five   six $")
        x = sp.next_match(cre_word, n="four")
        assert len(x) == 3
        assert x[0] == "one"
        assert x[1] == "two2"
        assert x[2] == "three"

        sp.reset()
        x = sp.next_match(cre_word, n="$")
        assert len(x) == 6
        assert x[-1] == "six"

    def test_closest(self):
        sp = parse("abc,def,ghi")
        assert sp.match_closest(("g", "h", "a", "b", "a")) == 2
        assert sp.match_closest(("z", "x", "y")) is None

    def test_closest2(self):
        sp = parse("AbC,dEf,gHI")
        assert sp.match_closest(("g", "h", "a", "b", "a")) == 2
        assert sp.match_closest(("z", "x", "y")) is None

    def test_matchAfter(self):
        sp = parse("param1 = value1, param2 value2")
        assert sp.match_after("param2", cre_var_name) == "value2"
        assert sp.match_after("param1", cre_var_name) == "value1"

    def test_intAfter(self):
        sp = parse("cows = 3 rabbits = 5 6")
        testing.assert_equal(sp.int_after("rabbits", n=2), (5, 6))
        assert sp.int_after("cows") == 3

    def test_floatAfter(self):
        sp = parse("value1 = 3; <value2> 3 4.5 5 6 </value2>")
        testing.assert_equal(sp.float_after("<value2>", n="</value2>"), (3, 4.5, 5, 6))
        assert sp.float_after("value1") == 3
