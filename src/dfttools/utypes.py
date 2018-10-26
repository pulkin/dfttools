"""
This submodule enhances `types.py` with units from
`numericalunits` package.
"""
import re
import numericalunits
import numpy
from numbers import Number

from . import types


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


class UnitsCollection(dict):
    """
    A collection of units.

    Example:

        >>> UnitsCollection(vectors="1/angstrom")
    """
    def get_nu_value(self, key, default=None):
        """
        Retrieves a `numerialunits` value by the key.
        Args:
            key (str): unit key;
            default: the default value;

        Returns:
            The numericalunits value.
        """
        if key in self and self[key] is not None:
            return eval_nu(self[key])
        else:
            return default

    def release(self, key, q):
        """
        Releases units from the given quantity.
        Args:
            key (str): unit key;
            q: the quantity;

        Returns:
            The quantity value measured in the given units.
        """
        return q / self.get_nu_value(key, default=1)

    def apply(self, key, q):
        """
        Applies units to the given quantity.
        Args:
            key (str): unit key;
            q: the quantity;

        Returns:
            The quantity value in "absolute" units.
        """
        return q * self.get_nu_value(key, default=1)


class UnitsMixin(object):
    """A core mixin to work with units."""
    def __init__(self, *args, **kwargs):
        if "units" in kwargs:
            if not isinstance(kwargs["units"], dict):
                raise ValueError("A dict expected for units kwarg, found {}".format(repr(kwargs["units"])))
            self.units = UnitsCollection(kwargs["units"])
            del kwargs["units"]
        else:
            self.units = UnitsCollection()
        super(UnitsMixin, self).__init__(*args, **kwargs)

    def __iter_fields_units__(self):
        for k in self.units:
            if k in dir(self):
                v = getattr(self, k)
                if isinstance(v, (numpy.ndarray, Number)):
                    yield k, v

    def __getstate_u__(self, state):
        return state

    def __getstate__(self):
        backup = {}

        # Release all units
        for k, v in self.__iter_fields_units__():
            backup[k] = v
            setattr(self, k, self.units.release(k, v))

        # Get the state
        state = self.__getstate_u__(super(UnitsMixin, self).__getstate__())
        if "units" in state:
            raise RuntimeError("The `units` key is reserved but was set by the parent class")
        state["units"] = dict(self.units)

        # Restore all data modified
        for k, v in backup.items():
            setattr(self, k, v)

        return state

    def __setstate_u__(self, state):
        pass

    def __setstate__(self, data):
        # Take units
        units = UnitsCollection(data["units"])
        del data["units"]
        # Init parent
        super(UnitsMixin, self).__setstate__(data)
        self.__setstate_u__(data)
        # Assign units
        self.units = units
        # Set all units
        for k, v in self.__iter_fields_units__():
            setattr(self, k, self.units.apply(k, v))


class Basis(UnitsMixin, types.Basis):
    """
    A units-aware version of the Basis.
    """


class UnitCell(UnitsMixin, types.UnitCell):
    """
    A units-aware version of the UnitCell.
    """

    def as_grid(self, fill=float("nan")):
        """
        Converts this UnitCell into a Grid.

        Kwargs:
            fill: default value to fill with;

        Returns:
            A new ``Grid``.
        """
        g = super(UnitCell, self).as_grid(fill=fill)
        return Grid(
            self,
            g.coordinates,
            g.values,
            units=self.units,
        )


class Grid(UnitsMixin, types.Grid):
    """
    A units-aware version of the Grid.
    """

    def as_unitCell(self):
        """
        Converts this Grid into a UnitCell.

        Returns:

            A new ``UnitCell``.
        """
        c = super(Grid, self).as_unitCell()
        return UnitCell(
            self,
            c.coordinates,
            c.values,
            units=self.units,
        )


class CrystalCell(UnitCell):
    """
    A unit cell of a crystal.
    """
    def __init__(self, *args, **kwargs):
        kw = dict(units=dict(vectors="angstrom"))
        kw.update(kwargs)
        super(CrystalCell, self).__init__(*args, **kw)


class BandsPath(UnitCell):
    """
    A band structure in a crystal.
    """
    def __init__(self, *args, **kwargs):
        kw = dict(units=dict(vectors="1/angstrom", values="eV", fermi="eV"), fermi=None)
        kw.update(kwargs)
        self.fermi = kw["fermi"]
        del kw["fermi"]
        super(BandsPath, self).__init__(*args, **kw)

    def as_grid(self, fill=float("nan")):
        """
        Converts this BandsPath into a BandsGrid.

        Kwargs:
            fill: default value to fill with;

        Returns:
            A new ``BandsGrid``.
        """
        g = super(BandsPath, self).as_grid(fill=fill)
        return BandsGrid(
            self,
            g.coordinates,
            g.values,
            fermi=self.fermi,
        )

    def __getstate_u__(self, state):
        state["fermi"] = self.fermi
        return state

    def __setstate_u__(self, state):
        self.fermi = state["fermi"]


class BandsGrid(Grid):
    """
    A band structure in a crystal.
    """
    def __init__(self, *args, **kwargs):
        kw = dict(units=dict(vectors="1/angstrom", values="eV", fermi="eV"), fermi=None)
        kw.update(kwargs)
        self.fermi = kw["fermi"]
        del kw["fermi"]
        super(BandsGrid, self).__init__(*args, **kw)

    def as_unitCell(self):
        """
        Converts this BandsGrid into a BandsPath.

        Returns:
            A new ``BandsPath``.
        """
        c = super(BandsGrid, self).as_unitCell()
        return BandsPath(
            self,
            c.coordinates,
            c.values,
            fermi=self.fermi,
        )

    def __getstate_u__(self, state):
        state["fermi"] = self.fermi
        return state

    def __setstate_u__(self, state):
        self.fermi = state["fermi"]
