"""
Contains helper routines to parse text.
"""
import re
import sys
    
import numpy

re_float = r"([-+]?[0-9]+\.?[0-9]*(?:[eEdD][-+]?[0-9]*)?)|(nan)"
re_int = r"([-+]?\d+)"
re_line = r"^(.*)$"
re_word = r"(\w+)"
re_varName = r"(\w[\d\w\(\)\._]+)"
re_nonspace = r"([^\s]+)"
re_quotedText = r"(\'.*?\')|(\".*?\")"

cre_float = re.compile(re_float)
cre_int = re.compile(re_int)
cre_line = re.compile(re_line,re.MULTILINE)
cre_word = re.compile(re_word)
cre_varName = re.compile(re_varName)
cre_nonspace = re.compile(re_nonspace)
cre_quotedText = re.compile(re_quotedText,re.DOTALL)

class ParseError(Exception):
    pass
    
class AbstractParser(object):
    
    def __init__(self, data):
        self.data = data
        self.parser = parse(data)
    
    @staticmethod
    def valid_header(header):
        """
        Checks whether the file header is an expected one. Used in
        automatic determination of file format.
        
        Args:
        
            header (str): the file header;
            
        Returns:
        
            True if the header is as expected.
        """
        raise NotImplementedError
        
    @staticmethod
    def valid_filename(name):
        """
        Checks whether the file name is an expected one. Used in
        automatic determination of file format.
        
        Args:
        
            name (str): the file name;
            
        Returns:
        
            True if the name is as expected.
        """
        raise NotImplementedError
    
class StringParser(object):
    """
    Simple parser for a string with position memory.
    
    This class can be used to parse words, numbers, floats and arrays
    from a given string. Based on
    `re <http://docs.python.org/2/library/re.html>`_, it provides the
    basic functionality for the rest of parsing libraries.
    
    Args:
    
        string (str): input string to be parsed.
        
    .. note::
        
        The input string can be further accessed by ``self.string``
        field. The contents of the string is not copied.
    
    """
    
    def __init__(self, string):
        self.string = string
        self.__position__ = 0
        self.__history__ = []
        
    def goto(self, expression, n = 1):
        """
        Goes to the beginning of nth occurrence of expression in the
        string.

        Args:
        
            expression (str,re.RegexObject): expression to match.
            If *expression* is str then the case is ignored.

        Kwargs:
        
            n (int): number of occurrences to match.
        
        Raises:
        
            StopIteration: No occurrences left in the string.
        """
        if isinstance(expression, str):
            expression = re.compile(re.escape(expression),re.I)
        ex_it = expression.finditer(self.string[self.__position__:])
        for i in range(n):
            start = next(ex_it).start()
        self.__position__ += start
        
    def pop(self):
        """
        Returns to the previously saved position of the parser.
        
        Raises:
        
            IndexError: No saved positions left.
        """
        self.__position__ = self.__history__.pop()
        
    def save(self):
        """
        Saves the current position of the parser.
        
        Example::
        
            sp = StringParser("A very important integer 123 describes something.")
            
            sp.skip("very") # The caret is set to the right of "very"
            sp.save() # The caret position is saved
            
            sp.skip("describes") # The caret is set to the right of "describes"
            # Now the call to StringParser.nextInt() will yield StopIteration.
            # To return the caret to the previously saved position
            # StringParser.pop() is used.
            sp.pop()
            
            # Now it is possible to read the integer
            sp.nextInt()
        """
        self.__history__.append(self.__position__)
        
    def skip(self, expression, n = 1):
        """
        Skips n occurrences of expression in the string.

        Args:
        
            expression (str,re.RegexObject): expression to match.
            If *expression* is str then the case is ignored.

        Kwargs:
        
            n (int): number of occurrences to skip.
           
        Raises:
        
            StopIteration: No occurrences left in the string.
        """
        if isinstance(expression, str):
            expression = re.compile(re.escape(expression),re.I)
        ex_it = expression.finditer(self.string[self.__position__:])
        for i in range(n):
            end = next(ex_it).end()
        self.__position__ += end
            
    def skipAll(self, expression):
        """
        Goes to the end of the last occurrence of a given expression in
        the string.

        Args:
        
            expression (str,re.RegexObject): expression to match.
            If *expression* is str then the case is ignored.
           
        Raises:
        
            StopIteration: No occurrences left in the string.
        """
        if isinstance(expression, str):
            expression = re.compile(re.escape(expression),re.I)
        ex_it = expression.finditer(self.string[self.__position__:])
        end = next(ex_it).end()
        while True:
            try:
                end = next(ex_it).end()
            except StopIteration:
                self.__position__ += end
                return
        
    def present(self, expression):
        """
        Test the string for the presence of expression.
        
        Args:
        
            expression (str,re.RegexObject): expression to match.
            If *expression* is str then the case is ignored.
        
        Returns:
        
            True if *expression* is matched to the right of current
            position of the caret.
        """
        try:
            self.distance(expression)
            return True
        except StopIteration:
            return False
                
    def distance(self, expression, n = 1, default = None):
        """
        Calculates distance to nth occurrence of expression in characters.
        
        Args:
        
            expression (str,re.RegexObject): expression to match. If
            *expression* is str then the case is ignored.

        Kwargs:
        
            n (int): consequetive number of expression to calculate
            distance to;
            
            default: return value if StopIteration occurs. Ignored if
            None.
        
        Returns:
        
            Numbers of characters between caret position and *nth*
            occurrence of *expression* or *default* if too few
            occurrences found.
            
        Raises:
        
            StopIteration: No occurrences left in the string.
        """
        if isinstance(expression, str):
            expression = re.compile(re.escape(expression),re.I)
        ex_it = expression.finditer(self.string[self.__position__:])
        try:
            for i in range(n):
                start = next(ex_it).start()
        except StopIteration:
            if default is None:
                raise
            else:
                return default
        return start
        
    def reset(self):
        """
        Resets the caret to the beginning of the string.
        """
        self.__position__ = 0
        
    def nextMatch(self, match, n = None):
        """
        Basic function for matching data.
        
        Args:
        
            match (re.RegexObject): object to match;
        
        Kwargs:
        
            n (array,int,str,re.RegexObject): specifies either shape of
            the numpy array returned or the regular expression to stop
            matching before;
                
        Returns:
        
            If *n* is specified returns a numpy array of a given shape
            filled with matches from string. Otherwise returns a single
            match. The caret is put behind the last match.
            
        Raises:
        
            StopIteration: Not enough matches left in the string.
                
        """
        ex_it = match.finditer(self.string[self.__position__:])
        if n is None:
            match = next(ex_it)
            result = match.group()
            self.__position__ += match.end()
            return result
        elif isinstance(n,(int, list, tuple, numpy.ndarray)):
            n_elements = n if isinstance(n,int) else numpy.prod(n)
            result = numpy.zeros(n_elements, dtype = object)
            if result.size > 0:
                for x in range(n_elements):
                    match = next(ex_it)
                    result[x] = match.group()
                self.__position__ += match.end()
            return result.reshape(n)
        else:
            lim = self.distance(n)
            result = []
            end = 0
            while True:
                try:
                    match = next(ex_it)
                    e = match.end()
                    if e<=lim:
                        result.append(match.group())
                        end = e
                    else:
                        break
                except StopIteration:
                    break
            self.__position__ += end
            return numpy.array(result, dtype = object)

    def matchAfter(self,after,match,n = None):
        """
        Matches pattern after another pattern and returns caret to initial
        position. Particularly useful for getting value for parameter
        name. Supports matching arrays via keyword parameter *n*.
        
        Args:
        
            after (re.RegexObject): pattern to skip;
            
            match (re.RegexObject): pattern to match;
            
        Kwargs:
        
            n (array,int,str,re.RegexObject): specifies either shape of
            the numpy array returned or the regular expression to stop
            matching before;
            
        Returns:
        
            If *n* is specified returns a numpy array of a given shape
            filled with matches from string. Otherwise returns a single
            match.

        Raises:
        
            StopIteration: Not enough matches left in the string.

        The function is equal to
        
            >>> sp = StringParser("Some string")
            >>> sp.save()
            >>> sp.skip(after)
            >>> result = sp.nextMatch(match, n = n)
            >>> sp.pop()
            
        """
        self.save()
        self.skip(after)
        result = self.nextMatch(match, n = n)
        self.pop()
        return result
        
    def nextInt(self, n = None):
        """
        Reads integers from string.
        
        Kwargs:
        
            n (array,int,str,re.RegexObject): specifies either shape of
            the numpy array returned or the regular expression to stop
            matching before;
                
        Returns:
        
            If *n* is specified returns a numpy array of a given shape
            filled with integers from string. Otherwise returns a single
            int. The caret is put behind the last integer read.
            
        Raises:
        
            StopIteration: Not enough integers left in the string.
            
        Example:
        
            >>> sp = StringParser("1 2 3 4 5 6 7 8 9 abc 10")
            >>> sp.nextInt((2,3))
            array([[1, 2, 3],
                [4, 5, 6]])
            >>> sp.nextInt("abc")
            array([ 7.,  8.,  9.])
                
        """
        result = self.nextMatch(cre_int, n = n)
        if n is None:
            return int(result)
        else:
            return result.astype(numpy.int)
            
    def intAfter(self, after, n = None):
        """
        Reads integers from string after the next regular expression.
        Returns the caret to initial position. Particularly useful for
        getting value for parameter name.
        
        Args:
        
            after (re.RegexObject) - pattern to skip;
            
        Kwargs:
        
            n (array,int,str,re.RegexObject): specifies either shape of
            the numpy array returned or the regular expression to stop
            matching before;
                
        Returns:
        
            If *n* is specified returns a numpy array of a given shape
            filled with integers from string. Otherwise returns a single
            int.
            
        Raises:
        
            StopIteration: Not enough integers left in the string.
            
        Example:
        
            >>> sp = StringParser("cows = 3, rabbits = 5")
            >>> sp.intAfter("rabbits")
            5
            >>> sp.intAfter("cows")
            3
                
        """
        self.save()
        self.skip(after)
        result = self.nextInt(n = n)
        self.pop()
        return result
        
    def nextFloat(self, n = None):
        """
        Reads floats from string.
        
        Kwargs:
        
            n (array,int,str,re.RegexObject): specifies either shape of
            the numpy array returned or the regular expression to stop
            matching before;
                
        Returns:
        
            If *n* is specified returns a numpy array of a given shape
            filled with floats from string. Otherwise returns a single
            float. The caret is put behind the last float read.
            
        Raises:
        
            StopIteration: Not enough floats left in the string.
            
        Example:
        
            >>> sp = StringParser("1.9 2.8 3.7 56.2E-2 abc")
            >>> sp.nextFloat(2)
            array([ 1.9, 2.8])
            >>> sp.nextFloat("abc")
            array([ 3.7  ,  0.562])
                
        """
        result = self.nextMatch(cre_float, n = n)
        if n is None:
            return float(result)
        else:
            return result.astype(numpy.float)
            
    def floatAfter(self, after, n = None):
        """
        Reads floats from string after the next regular expression.
        Returns the caret to initial position. Particularly useful for
        getting value for parameter name.
        
        Args:
        
            after (re.RegexObject) - pattern to skip;
        
        Kwargs:
        
            n (array,int,str,re.RegexObject): specifies either shape of
            the numpy array returned or the regular expression to stop
            matching before;
                
        Returns:
        
            If *n* is specified returns a numpy array of a given shape
            filled with floats from string. Otherwise returns a single
            float.
            
        Raises:
        
            StopIteration: Not enough floats left in the string.
            
        Example:
        
            >>> sp = StringParser("apples = 3.4; bananas = 7")
            >>> sp.floatAfter("bananas")
            7.0
            >>> sp.floatAfter("apples")
            3.4
                
        """
        self.save()
        self.skip(after)
        result = self.nextFloat(n = n)
        self.pop()
        return result

    def nextLine(self, n = None):
        """
        Reads lines from string.
        
        Kwargs:
        
            n (array,int,str,re.RegexObject): specifies either shape of
            the numpy array returned or the regular expression to stop
            matching before;
                
        Returns:
        
            If *n* is specified returns a numpy array of a given shape
            filled with lines from string. Otherwise returns a single
            line. The caret is put behind the last line read.
            
        Raises:
        
            StopIteration: Not enough lines left in the string.
            
        """
        if self.__position__ == len(self.string):
            raise StopIteration
        result = self.nextMatch(cre_line, n = n)
        if self.__position__ < len(self.string):
            self.__position__ += 1
        return result
        
    def startOfLine(self):
        """
        Goes to the beginning of the current line.
        """
        if self.__position__ > 0:
            self.__position__ -= 1
        else:
            return
            
        while not self.string[self.__position__] == "\n":
            if self.__position__ == 0:
                return
            self.__position__ -= 1
            
        self.__position__ += 1
        
    def closest(self, exprs):
        """
        Returns the closest match of a set of expressions.
        
        Args:
        
            exprs (list): a set of expressions being matched.
            
        Returns:
        
            Index of the closest expression. The distance is measured to
            the beginnings of matches. Returns None if none of
            expressions matched.
        
        Example:
        
            >>> sp = StringParser("This is a large string")
            >>> sp.closest(("a","string","this"))
            2
            
        """
        patterns = tuple(re.escape(i) if isinstance(i,str) else i.pattern for i in exprs)
        match = re.search("("+(")|(".join(patterns))+")",self.string[self.__position__:],re.I)
        if match is None:
            return None
        else:
            matchedString = match.group()
            for i in range(len(patterns)):
                if not re.search(patterns[i],matchedString,re.I) is None:
                    return i

parse = StringParser
