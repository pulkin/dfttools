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
                    # if not isinstance(target, util.array):
                    setattr(self, k, util.array(target, units=v))


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
    """
    def __init__(self, *args, **kwargs):
        self.fermi = kwargs.pop("fermi", None)
        super(FermiMixin, self).__init__(*args, **kwargs)

    def __getstate__(self):
        state = super(FermiMixin, self).__getstate__()
        state["fermi"] = self.fermi.copy()
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
        self.__fermi__ = util.array(v, units=self.default_units.get("fermi", None))


class BandsPath(FermiMixin, UnitsMixin, types.UnitCell):
    """
    A band structure in a crystal.
    """

    default_units = dict(vectors="1/angstrom", values="eV", fermi="eV")

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


class BandsGrid(FermiMixin, UnitsMixin, types.Grid):
    """
    A band structure in a crystal.
    """

    default_units = dict(vectors="1/angstrom", values="eV", fermi="eV")

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
