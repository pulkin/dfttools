"""
This submodule enhances `ndarray` with units from `numericalunits` package.
"""
import json

import numericalunits
import numpy
import ast
import operator as op


def eval_nu(s):
    """
    Evaluates a numericalunits expression.

    Args:
        s (str): expression to evaluate;

    Returns:
        The result of the evaluation.
    """
    operators = {ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}

    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
            return operators[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.Name):
            return getattr(numericalunits, node.id)
        else:
            raise TypeError(node)

    return _eval(ast.parse(s, mode='eval').body)


def invert_nu(s):
    """
    Inverts a numericalunits expression.
    Args:
        s (str): expression to invert;

    Returns:
        The inverted expression.
    """
    return "1/(" + s + ")"


def array_to_json(a):
    """
    Creates a JSON-compatible representation of an array.

    Args:
        a (ndarray): a numpy array;

    Returns:
        JSON representation of the array.
    """
    if not isinstance(a, array):
        a = numpy.asarray(a).view(array)
    if a.units is None:
        a = a.view()
    else:
        a = a / eval_nu(a.units)
    is_complex = numpy.iscomplexobj(a)
    if is_complex:
        _a = numpy.concatenate((a.real[..., numpy.newaxis], a.imag[..., numpy.newaxis]), axis=-1)
    else:
        _a = a
    return dict(
        _type="numpy",
        data=_a.tolist(),
        complex=is_complex,
        units=a.units,
    )


class ArrayWithUnits(numpy.ndarray):
    """
    Enhances numpy array with units from `numericalunits`.
    """

    def __new__(cls, *args, **kwargs):
        units = kwargs.pop("units", None)
        obj = numpy.asarray(*args, **kwargs).view(cls)
        obj.units = units
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.units = getattr(obj, 'units', None)

    to_json = array_to_json

    @classmethod
    def from_json(cls, data):
        """
        Recovers the array from JSON representation.

        Args:
            data (dict): the JSON representation;

        Returns:
            The array recovered.
        """
        if not isinstance(data, dict):
            raise TypeError("Dict expected, found: {}".format(repr(data)))
        if "_type" not in data:
            raise TypeError("Missing the '_type' key in the representation")
        if data["_type"] != "numpy":
            raise TypeError("The data type is wrong: found {}, expected 'numpy'".format(data["_type"]))

        a = numpy.asarray(data["data"])
        if data["complex"]:
            a = a[..., 0] + 1j * a[..., 1]

        if data["units"] is not None:
            a *= eval_nu(data["units"])

        return cls(a, units=data["units"])

    def __reduce__(self):
        if self.units is None:
            state = super(ArrayWithUnits, self).__reduce__()
        else:
            state = numpy.ndarray.__reduce__(self / eval_nu(self.units))
        return state[:2] + (state[2] + (self.units,),)

    def __setstate__(self, state):
        super(ArrayWithUnits, self).__setstate__(state[:-1])
        self.units = state[-1]
        if self.units is not None:
            self *= eval_nu(self.units)


array = ArrayWithUnits


def eV(x):
    """A shortcut to adding units to values in eV."""
    return array(x, units="eV")


def angstrom(x):
    """A shortcut to adding units to values in angstrom."""
    return array(x, units="angstrom")


def inv_angstrom(x):
    """A shortcut to adding units to values in angstrom."""
    return array(x, units="1/angstrom")


def eV_angstrom(x):
    """A shortcut to adding units to atomic forces."""
    return array(x, units="eV/angstrom")


def K(x):
    """A shortcut to adding Kelvin units to temperature."""
    return array(x, units="K")


def cast_units(destination, source, inv=False):
    """
    Casts units from one array to another.

    Args:
        destination (ndarray): destination array;
        source (ndarray): array to cast units from;
        inv (bool): whether to cast inverse units;

    Returns:
        A new array with data from `destination`
        and units from `source`.
    """
    if not isinstance(source, array) or source.units is None:
        return destination

    if not inv:
        return array(destination, units=source.units)
    else:
        return array(destination, units=invert_nu(source.units))


class JSONEncoderWithArray(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, numpy.ndarray):
            o = o.view(array)
            return array_to_json(o)
        else:
            return super(JSONEncoderWithArray, self).default(o)


def object_hook(d):
    if "_type" in d:
        return array.from_json(d)
    else:
        return d


# Shortcuts for dumping/loading jsons with arrays
def dump(*args, **kwargs):
    kwargs["cls"] = JSONEncoderWithArray
    return json.dump(*args, **kwargs)


def dumps(*args, **kwargs):
    kwargs["cls"] = JSONEncoderWithArray
    return json.dumps(*args, **kwargs)


def load(*args, **kwargs):
    kwargs["object_hook"] = object_hook
    return json.load(*args, **kwargs)


def loads(*args, **kwargs):
    kwargs["object_hook"] = object_hook
    return json.loads(*args, **kwargs)
