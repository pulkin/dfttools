"""
This submodule enhances `ndarray` with units from `numericalunits` package.
"""
import re
import json

import numericalunits
import numpy


def eval_nu(s):
    """
    Evaluates numericalunits expression.
    Args:
        s (str): en expression to evaluate;

    Returns:
        The result of evaluation.
    """
    match = re.match(r'\s*\w*((\s*[*/]\s*\w*)*)\s*$', s)
    if match is None:
        raise ValueError("Not a valid numericalunits expression: {}".format(s))

    result = 1.

    for i in re.finditer(r"([*/])\s*(\w*)", "*" + s):
        op, name = i.groups()
        name = str(name)
        if name == "1":
            val = 1.
        elif name in dir(numericalunits):
            val = getattr(numericalunits, name)
        else:
            raise ValueError("'{}' not found in numericalunits".format(name))

        if op == "*":
            result *= val
        else:
            result /= val

    return result


def array_to_json(a):
    """
    Creates a JSON-compatible representation of an array.
    Returns:
        The JSON representation.
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
        Builds an array from JSON representation.
        Args:
            data (dict): the JSON representation;

        Returns:
            The array.
        """
        if not isinstance(data, dict):
            raise TypeError("Dict expected, found: {}".format(repr(data)))
        if "_type" not in data:
            raise TypeError("Missing the '_type' key in the reresentation")
        if data["_type"] != "numpy":
            raise TypeError("The data tpye is wrong: found {}, expected 'numpy'".format(data["_type"]))

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


def cast_units(destination, source):
    """
    Casts units from one array to another.
    Args:
        destination (ndarray): destination array;
        source (ndarray): array to cast units from;

    Returns:
        If `a2` is `ArrayWithUnits` casts units from `a2` into `a1`, otherwise returns `a1`.
    """
    return destination if not isinstance(source, array) else array(destination, units=source.units)


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
