import unittest
import re
import math

from numpy import testing

from dfttools.parsers.generic import parse, cre_varName, cre_word

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
        assert sp.__position__ == len(sp.string)-6
        sp.__position__ += 1
        self.assertRaises(StopIteration,sp.goto,"this")
        
    def test_goto_1(self):
        
        sp = parse("ABC, Abc, abc, aBc, aBC")
        sp.goto("abC")
        assert sp.__position__ == 0
        sp.goto("aBc")
        assert sp.__position__ == 0
        sp.goto(re.compile("aBc"))
        assert sp.__position__ == 15
        self.assertRaises(StopIteration,sp.goto,"bac")
        
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
        
    def test_skip(self):
        
        sp = parse("ABC, Abc, abc, aBc, aBC, aBC, aBC")
        sp.skip("abc")
        assert sp.__position__ == 3
        sp.skip("abc", n=2)
        assert sp.__position__ == 13
        sp.skip(re.compile("aBC"), n=2)
        assert sp.__position__ == 28
        self.assertRaises(StopIteration,sp.skip,re.compile("Abc"))
        
    def test_skipAll(self):
        
        sp = parse("ABC, ABC, abc, aBc, aBC, aBC, aBC, def")
        self.assertRaises(StopIteration,sp.skipAll,"xyz")
        sp.skipAll(re.compile("ABC"))
        assert sp.__position__ == 8
        sp.skipAll("abc")
        assert sp.__position__ == 33
        self.assertRaises(StopIteration,sp.skipAll,"abc")
        
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
        assert sp.distance("abc", n = 2) == 12
        self.assertRaises(StopIteration,sp.distance,"abc",n=3)
        assert sp.distance("abc",n=4,default = -1)==-1
        sp.__position__ = 1
        assert sp.distance("abc") == 11
        assert sp.distance("abc", default = -1) == 11
        
    def test_nextInt(self):
        
        sp = parse("123abc456 78.8 +45.68-73 1 2 3 4 5 6 7 8 abc 9")
        assert sp.nextInt() == 123
        assert sp.nextInt() == 456
        assert sp.nextInt() == 78
        assert sp.nextInt() == 8
        testing.assert_equal(sp.nextInt(3),(45,68,-73))
        testing.assert_equal(sp.nextInt((2,3)),((1,2,3),(4,5,6)))
        testing.assert_equal(sp.nextInt("abc"),(7,8))
        assert sp.nextInt() == 9
        self.assertRaises(StopIteration,sp.nextInt)
        
    def test_nextFloat(self):
        sp = parse("123, 12a3, nan, 5, nan, 4.66, 12., 356.2, 4e-2, 4.e-5, 56.2E-2 -36.6 1.0 2.0 3.0 abc 4.0")
        assert sp.nextFloat() == 123
        testing.assert_equal(sp.nextFloat(2),(12,3))
        assert math.isnan(sp.nextFloat())
        testing.assert_equal(sp.nextFloat(2),(5,float("nan")))
        testing.assert_equal(sp.nextFloat((2,3)),((4.66,12.,356.2),(4e-2,4e-5,56.2e-2)))
        assert sp.nextFloat() == -36.6
        testing.assert_equal(sp.nextFloat("abc"),(1.,2.,3.))
        assert sp.nextFloat() == 4.0        
        self.assertRaises(StopIteration,sp.nextFloat)
        
    def test_nextFloat2(self):
        
        sp = parse("123, 456; 789")
        testing.assert_equal(sp.nextFloat(";"),(123, 456))
        
    def test_lineOperations(self):
        
        sp = parse("\none\ntwo\nthree\nfour")
        sp.startOfLine()
        assert sp.__position__ == 0
        assert sp.nextLine() == ""
        assert sp.__position__ == 1
        sp.skip("two")
        sp.startOfLine()
        assert sp.__position__ == 5
        testing.assert_equal(sp.nextLine(2), ("two","three"))
        assert sp.__position__ == 15
        sp.startOfLine()
        assert sp.__position__ == 15
        sp.skip("four")
        sp.startOfLine()
        assert sp.__position__ == 15
        assert sp.nextLine() == "four"
        self.assertRaises(StopIteration,sp.nextLine)
        
    def test_lineOperations2(self):
        
        sp = parse("Hello\n World!")
        sp.__position__ = 2
        sp.startOfLine()
        assert sp.__position__ == 0
        
        sp.__position__ = 10
        sp.startOfLine()
        assert sp.__position__ == 6
        
    def test_nextMatch(self):
        
        sp = parse("one two2 three, four\n  five   six seven")
        assert sp.nextMatch(cre_word) == "one"
        assert sp.__position__ == 3
        assert sp.nextMatch(cre_word) == "two2"
        assert sp.__position__ == 8
        assert sp.nextMatch(cre_word) == "three"
        assert sp.nextMatch(cre_word) == "four"
        assert sp.nextMatch(cre_word) == "five"
        assert sp.nextMatch(cre_word) == "six"
        assert sp.nextMatch(cre_word) == "seven"
        self.assertRaises(StopIteration,sp.nextMatch,cre_word)
        
    def test_nextMatch2(self):
        
        sp = parse("one two2 three, four\n  five   six $")
        x = sp.nextMatch(cre_word, n="four")
        assert len(x) == 3
        assert x[0] == "one"
        assert x[1] == "two2"
        assert x[2] == "three"
        
        sp.reset()
        x = sp.nextMatch(cre_word, n="$")
        assert len(x) == 6
        assert x[-1] == "six"
        
    def test_closest(self):
        
        sp = parse("abc,def,ghi")
        assert sp.closest(("g","h","a","b","a")) == 2
        assert sp.closest(("z","x","y")) is None
        
    def test_closest2(self):
        
        sp = parse("AbC,dEf,gHI")
        assert sp.closest(("g","h","a","b","a")) == 2
        assert sp.closest(("z","x","y")) is None
        
    def test_matchAfter(self):
        
        sp = parse("param1 = value1, param2 value2")
        assert sp.matchAfter("param2",cre_varName)=="value2"
        assert sp.matchAfter("param1",cre_varName)=="value1"
        
    def test_intAfter(self):
        
        sp = parse("cows = 3 rabbits = 5 6")
        testing.assert_equal(sp.intAfter("rabbits",n = 2),(5, 6))
        assert sp.intAfter("cows")==3
        
    def test_floatAfter(self):
        
        sp = parse("value1 = 3; <value2> 3 4.5 5 6 </value2>")
        testing.assert_equal(sp.floatAfter("<value2>", n = "</value2>"),(3, 4.5, 5, 6))
        assert sp.floatAfter("value1") == 3
