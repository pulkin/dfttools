"""
This submodule enhances `types.py` with units-aware arrays.
"""
import numpy
from numbers import Number

from . import types, util


class UnitsMixin(object):
    """
    A core mixin to work with units.
    """

    default_units = {}

    def __init__(self, *args, **kwargs):
        super(UnitsMixin, self).__init__(*args, **kwargs)
        for k, v in self.default_units.items():
            if k in dir(self):
                target = getattr(self, k)
                if isinstance(target, (numpy.ndarray, Number)):
                    if not isinstance(target, util.array):
                        setattr(self, k, util.array(target, units=v))


# TODO: remove most of these classes

class RealSpaceBasis(UnitsMixin, types.Basis):
    """
    Basis in real space.
    """

    default_units = dict(vectors="angstrom")


class ReciprocalSpaceBasis(UnitsMixin, types.Basis):
    """
    Basis in reciprocal space.
    """

    default_units = dict(vectors="1/angstrom")


class CrystalCell(UnitsMixin, types.UnitCell):
    """
    A unit cell of a crystal.
    """

    default_units = dict(vectors="angstrom", values=None)


class CrystalGrid(UnitsMixin, types.Grid):
    """
    A grid in real space.
    """

    default_units = dict(vectors="angstrom", values=None)


class FermiMixin(object):
    """
    A mixin to add the Fermi attribute.

    Args:
        fermi (float): the Fermi level value;
    """
    def __init__(self, *args, **kwargs):
        self.fermi = kwargs.pop("fermi", None)
        super(FermiMixin, self).__init__(*args, **kwargs)

    def __getstate__(self):
        state = super(FermiMixin, self).__getstate__()
        state["fermi"] = self.fermi.copy() if self.fermi is not None else None
        return state

    def __setstate__(self, state):
        fermi = state.pop("fermi")
        super(FermiMixin, self).__setstate__(state)
        self.fermi = fermi

    @property
    def fermi(self):
        return self.__fermi__

    @fermi.setter
    def fermi(self, v):
        if isinstance(v, util.ArrayWithUnits):
            self.__fermi__ = v.copy()
        elif isinstance(v, Number):
            self.__fermi__ = util.array(v, units=self.default_units.get("fermi", None))
        elif v is None:
            self.__fermi__ = None
        else:
            raise ValueError("Only numeric values or None are accepted for the Fermi, found: {}".format(repr(v)))


class BandsPath(FermiMixin, UnitsMixin, types.UnitCell):
    """
     A class describing a band structure along a path.
     See `dfttools.types.UnitCell` and `dfttools.utypes.FermiMixin` for arguments.
     """

    default_units = dict(vectors="1/angstrom", values="eV", fermi="eV")

    def as_grid(self, fill=float("nan")):
        g = super(BandsPath, self).as_grid(fill=fill)
        return BandsGrid(
            self,
            g.coordinates,
            g.values,
            fermi=self.fermi,
        )
    as_grid.__doc__ = types.UnitCell.as_grid.__doc__

    def interpolate(self, *args, **kwargs):
        result = super(BandsPath, self).interpolate(*args, **kwargs)
        return BandsPath(result.vectors, result.coordinates, result.values, meta=result.meta, fermi=self.fermi)
    interpolate.__doc__ = types.UnitCell.interpolate.__doc__


class BandsGrid(FermiMixin, UnitsMixin, types.Grid):
    """
     A class describing a band structure on a grid.
     See `dfttools.types.Grid` and `dfttools.utypes.FermiMixin` for arguments.
    """

    default_units = dict(vectors="1/angstrom", values="eV", fermi="eV")

    def as_cell(self):
        c = super(BandsGrid, self).as_cell()
        return BandsPath(
            self,
            c.coordinates,
            c.values,
            fermi=self.fermi,
        )
    as_cell.__doc__ = types.Grid.as_cell.__doc__

    def interpolate_to_cell(self, *args, **kwargs):
        result = super(BandsGrid, self).interpolate_to_cell(*args, **kwargs)
        return BandsPath(result.vectors, result.coordinates, result.values, meta=result.meta, fermi=self.fermi)
    interpolate_to_cell.__doc__ = types.Grid.interpolate_to_cell.__doc__

    def interpolate_to_path(self, *args, **kwargs):
        result = super(BandsGrid, self).interpolate_to_path(*args, **kwargs)
        return BandsPath(result.vectors, result.coordinates, result.values, meta=result.meta, fermi=self.fermi)
    interpolate_to_path.__doc__ = types.Grid.interpolate_to_path.__doc__

    def interpolate_to_grid(self, *args, **kwargs):
        result = super(BandsGrid, self).interpolate_to_grid(*args, **kwargs)
        return BandsGrid(result.vectors, result.coordinates, result.values, meta=result.meta, fermi=self.fermi)
    interpolate_to_grid.__doc__ = types.Grid.interpolate_to_grid.__doc__
