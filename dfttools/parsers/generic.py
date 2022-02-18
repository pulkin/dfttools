"""
Contains helper routines to parse text.
"""
import json
import re

import numpy

re_float = r"([-+]?[0-9]+\.?[0-9]*(?:[eEdD][-+]?[0-9]*)?)|(nan)"
re_int = r"([-+]?\d+)"
re_line = r"^(.*)$"
re_word = r"(\w+)"
re_var_name = r"(\w[\d\w\(\)\._]+)"
re_non_space = r"([^\s]+)"
re_quotedText = r"(\'.*?\')|(\".*?\")"

cre_float = re.compile(re_float)
cre_int = re.compile(re_int)
cre_line = re.compile(re_line, re.MULTILINE)
cre_word = re.compile(re_word)
cre_var_name = re.compile(re_var_name)
cre_non_space = re.compile(re_non_space)
cre_quotedText = re.compile(re_quotedText, re.DOTALL)


class ParseError(Exception):
    pass


class IdentifiableParser(object):
    """
    A stub for those parsers which can be identified by the content.
    """
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


class AbstractTextParser(object):
    """
    A root class for text parsers.
    
    Args:
        f (str, file): text to parse or a file to read;
    """

    def __init__(self, f):
        if hasattr(f, "read"):
            self.file = f
            self.data = f.read()
        else:
            self.file = None
            self.data = f
        self.parser = parse(self.data)

    def __collect_source_meta__(self):
        meta = {}
        if self.file is not None:
            meta["source-file-name"] = self.file.name
        return meta


class AbstractJSONParser(object):
    """
    A root class for JSON parsers.
    
    Args:
        data (str, dict, file): JSON data to parse;
    """
    loads = staticmethod(json.loads)

    def __init__(self, data):
        if hasattr(data, "read"):
            data = data.read()
        if isinstance(data, dict):
            self.json = data
        else:
            self.json = self.loads(data)

    def __set_units__(self, field, units):
        self.json[field] = numpy.array(self.json[field]) * units

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
        return name.endswith(".json")


class StringParser(object):
    """
    Simple parser for a string with position memory.
    
    This class can be used to parse words, numbers, floats and arrays
    from the given string. Based on
    `re <http://docs.python.org/2/library/re.html>`_, it provides the
    basic functionality for the rest of parsing libraries.
    
    Args:
        string (str): input string to be parsed;
        
    .. note::
        The input string can be further accessed by `self.string`
        field. The contents of the string is not copied.
    
    """

    def __init__(self, string):
        self.string = string
        self.__position__ = 0
        self.__history__ = []

    def goto(self, expression, n=1):
        """
        Goes to the beginning of nth occurrence of expression in the
        string.

        Args:
            expression (str, re.RegexObject): expression to match.
            If `expression` is str then the case is ignored;
            n (int): the number of matches to skip (minus one);

        Raises:
            StopIteration: If no occurrences left in the string.
        """
        if isinstance(expression, str):
            expression = re.compile(re.escape(expression), re.I)
        ex_it = expression.finditer(self.string[self.__position__:])
        start = 0
        for i in range(n):
            start = next(ex_it).start()
        self.__position__ += start

    def pop(self):
        """
        Returns to the previously saved position of the parser.
        
        Raises:
            IndexError: If no saved positions left.
        """
        self.__position__ = self.__history__.pop()

    def save(self):
        """
        Saves the current position of the parser.
        
        Example::
            >>> sp = StringParser("A very important integer 123 describes something.")
            >>> sp.skip("very") # The caret is set to the right of "very"
            >>> sp.save() # The caret position is saved
            >>> sp.skip("describes") # The caret is set to the right of "describes"
            >>> sp.pop() # The caret position is returned back to the right of "very"
            >>> sp.next_int() # Now it is possible to read the integer
        """
        self.__history__.append(self.__position__)

    def fw(self, i):
        """
        Jumps forward by the given number of symbols.
        Args:
            i (int): the number of symbols to skip;
        """
        if self.__position__ + i > len(self.string):
            raise ValueError("Hops beyond the end of the data")
        self.__position__ += i

    def skip(self, expression, n=1):
        """
        Skips n occurrences of expression in the string.

        Args:
            expression (str, re.RegexObject): expression to match.
            If `expression` is str then the case is ignored;
            n (int): the number of occurrences to skip;
           
        Raises:
            StopIteration: If no occurrences left in the string.
        """
        if isinstance(expression, str):
            expression = re.compile(re.escape(expression), re.I)
        ex_it = expression.finditer(self.string[self.__position__:])
        end = 0
        for i in range(n):
            end = next(ex_it).end()
        self.__position__ += end

    def skip_all(self, expression):
        """
        Goes to the end of the last occurrence of the given expression in
        the string.

        Args:
            expression (str, re.RegexObject): the expression to match.
            If `expression` is str then the case is ignored;
           
        Raises:
            StopIteration: If no occurrences left in the string.
        """
        if isinstance(expression, str):
            expression = re.compile(re.escape(expression), re.I)
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
        Test the string for the presence of an expression.
        
        Args:
            expression (str, re.RegexObject): the expression to match.
            If `expression` is str then the case is ignored;
        
        Returns:
            True if `expression` is matched to the right of current
            position of the caret.
        """
        try:
            self.distance(expression)
            return True
        except StopIteration:
            return False

    def distance(self, expression, n=1, to="head", default=None):
        """
        Calculates the distance to nth occurrence of expression in characters.
        
        Args:
            expression (str, re.RegexObject): the expression to match. If
            `expression` is str then the case is ignored;
            n (int): the number of occurrences to skip (minus one);
            to (str): "head" or "tail": whether measure distance to head or tail;
            default: the return value if StopIteration occurs. Raises if
            `None`;
        
        Returns:
            The number of characters between the caret position and the
            `n`th occurrence of `expression`.
            
        Raises:
            StopIteration: If no occurrences left in the string.
        """
        if isinstance(expression, str):
            expression = re.compile(re.escape(expression), re.I)
        ex_it = expression.finditer(self.string[self.__position__:])
        try:
            result = 0
            for i in range(n):
                token = next(ex_it)
                result = token.start() if to == "head" else token.end()
            return result
        except StopIteration:
            if default is None:
                raise
            else:
                return default

    def reset(self):
        """
        Resets the caret to the beginning of the string.
        """
        self.__position__ = 0

    def next_match(self, match, n=None):
        """
        Basic function for matching patterns.
        
        Args:
            match (re.RegexObject): the regex to match;
            n (array, int, str, re.RegexObject): specifies either shape of
            the numpy array returned or the regular expression to stop
            matching before;
                
        Returns:
            If `n` is specified returns a numpy array of the given shape
            filled with matches from the string. Otherwise returns a single
            match. The caret is put behind the last match.
            
        Raises:
            StopIteration: If not enough matches left in the string.
        """
        ex_it = match.finditer(self.string[self.__position__:])
        if n is None:
            match = next(ex_it)
            result = match.group()
            self.__position__ += match.end()
            return result
        elif isinstance(n, (int, list, tuple, numpy.ndarray)):
            n_elements = n if isinstance(n, int) else numpy.prod(n)
            result = numpy.zeros(n_elements, dtype=object)
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
                    if e <= lim:
                        result.append(match.group())
                        end = e
                    else:
                        break
                except StopIteration:
                    break
            self.__position__ += end
            return numpy.array(result, dtype=object)

    def match_after(self, after, match, n=None):
        """
        Matches pattern after another pattern and returns caret to the
        initial position. Particularly useful for getting value for
        parameter name. Supports matching arrays via keyword parameter `n`.
        
        Args:
            after (re.RegexObject): regex to skip;
            match (re.RegexObject): regex to match;
            n (array, int, str, re.RegexObject): specifies either shape of
            the numpy array returned or the regular expression to stop
            matching before;
            
        Returns:
            If `n` is specified returns a numpy array of the given shape
            filled with matches from the string. Otherwise returns a single
            match.

        Raises:
            StopIteration: Not enough matches left in the string.

        The function is equal to
            >>> sp = StringParser("Some string")
            >>> sp.save()
            >>> sp.skip(after)
            >>> result = sp.next_match(match, n=n)
            >>> sp.pop()
            
        """
        self.save()
        self.skip(after)
        result = self.next_match(match, n=n)
        self.pop()
        return result

    def next_int(self, n=None):
        """
        Reads integers from string.
        
        Kwargs:
            n (array, int, str, re.RegexObject): specifies either the shape
            of the numpy array returned or the regular expression to stop
            matching before;
                
        Returns:
            If `n` is specified returns a numpy array of the given shape
            filled with integers from the string. Otherwise returns a single
            int. The caret is put behind the last integer read.
            
        Raises:
            StopIteration: Not enough integers left in the string.
            
        Example:
            >>> sp = StringParser("1 2 3 4 5 6 7 8 9 abc 10")
            >>> sp.next_int((2,3))
            array([[1, 2, 3],
                [4, 5, 6]])
            >>> sp.next_int("abc")
            array([ 7.,  8.,  9.])  
        """
        result = self.next_match(cre_int, n=n)
        if n is None:
            return int(result)
        else:
            return result.astype(numpy.int)

    def int_after(self, after, n=None):
        """
        Reads integers from string after the next regular expression
        match. Returns the caret to the initial position after matching
        the int. Particularly useful for parsing parameter-value pairs.
        
        Args:
            after (re.RegexObject): the pattern to skip;
            n (array, int, str, re.RegexObject): specifies either the
            shape of the numpy array returned or the regular expression
            to stop matching before;
                
        Returns:
            If `n` is specified returns a numpy array of the given shape
            filled with integers from the string. Otherwise returns a single
            int.
            
        Raises:
            StopIteration: If not enough integers left in the string.
            
        Example:
            >>> sp = StringParser("cows = 3, rabbits = 5")
            >>> sp.int_after("rabbits")
            5
            >>> sp.int_after("cows")
            3
        """
        self.save()
        self.skip(after)
        result = self.next_int(n=n)
        self.pop()
        return result

    def next_float(self, n=None):
        """
        Reads floats from string.

        Kwargs:
            n (array, int, str, re.RegexObject): specifies either the shape
            of the numpy array returned or the regular expression to stop
            matching before;

        Returns:
            If `n` is specified returns a numpy array of the given shape
            filled with floats from the string. Otherwise returns a single
            float. The caret is put behind the last float read.

        Raises:
            StopIteration: Not enough floats left in the string.

        Example:
            >>> sp = StringParser("1.9 2.8 3.7 56.2E-2 abc")
            >>> sp.next_float(2)
            array([ 1.9, 2.8])
            >>> sp.next_float("abc")
            array([ 3.7  ,  0.562])
                
        """
        result = self.next_match(cre_float, n=n)
        if n is None:
            return float(result.replace('d', 'e').replace('D', 'E'))
        else:
            return result.astype(numpy.float)

    def float_after(self, after, n=None):
        """
        Reads floats from string after the next regular expression
        match. Returns the caret to the initial position after matching
        the float. Particularly useful for parsing parameter-value pairs.
        
        Args:
            after (re.RegexObject): the pattern to skip;
            n (array, int, str, re.RegexObject): specifies either the
            shape of the numpy array returned or the regular expression
            to stop matching before;
                
        Returns:
            If `n` is specified returns a numpy array of the given shape
            filled with floats from the string. Otherwise returns a single
            float.
            
        Raises:
            StopIteration: If not enough floats left in the string.
            
        Example:
            >>> sp = StringParser("apples = 3.4; bananas = 7")
            >>> sp.float_after("bananas")
            7.0
            >>> sp.float_after("apples")
            3.4
                
        """
        self.save()
        self.skip(after)
        result = self.next_float(n=n)
        self.pop()
        return result

    def next_line(self, n=None):
        """
        Reads lines from string.
        
        Args:
            n (array, int, str, re.RegexObject): specifies either the shape
            of the numpy array returned or the regular expression to stop
            matching before;
                
        Returns:
            If `n` is specified returns a numpy array of the given shape
            filled with lines from the string. Otherwise returns a single
            line of text. The caret is put behind the last line read.
            
        Raises:
            StopIteration: If not enough lines left in the string.
            
        """
        if self.__position__ == len(self.string):
            raise StopIteration
        result = self.next_match(cre_line, n=n)
        if self.__position__ < len(self.string):
            self.__position__ += 1
        return result

    def rtn(self):
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

    def match_closest(self, expressions):
        """
        Returns the closest match of several expressions.
        
        Args:
            expressions (list, tuple): a set of expressions being matched.
            
        Returns:
            The index of the closest expression. The distance is measured to
            the beginnings of each match. Returns `None` if none of the
            expressions matched.
        
        Example:
            >>> sp = StringParser("This is a large string")
            >>> sp.match_closest(("a","string","this"))
            2
        """
        patterns = tuple(re.escape(i) if isinstance(i, str) else i.pattern for i in expressions)
        match = re.search("(" + (")|(".join(patterns)) + ")", self.string[self.__position__:], re.I)
        if match is None:
            return None
        else:
            matched_string = match.group()
            for i in range(len(patterns)):
                if re.search("^" + patterns[i] + "$", matched_string, re.I) is not None:
                    return i


parse = StringParser
