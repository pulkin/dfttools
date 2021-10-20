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


class FermiUndefinedException(Exception):
    pass


class BandsMissingException(Exception):
    pass


class FermiCrossingException(Exception):
    pass


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

    @property
    def nocc(self):
        """The number of occupied bands."""
        if self.fermi is None:
            raise FermiUndefinedException("The Fermi level is not defined")
        nocc = (self.values < self.fermi).sum(axis=-1).reshape(-1)
        if not numpy.all(nocc == nocc[0]):
            raise FermiCrossingException("Is a metal")
        return nocc[0]

    @property
    def nvirt(self):
        """Number of unoccupied (virtual) states."""
        return self.values.shape[-1] - self.nocc

    @property
    def gapped(self):
        """Checks if it is a gapped band structure."""
        try:
            self.nocc
            return True
        except FermiCrossingException as e:
            return False

    @property
    def vbt(self):
        """Valence band maximum (top) value."""
        nocc = self.nocc
        if nocc == 0:
            raise BandsMissingException("No valence bands")
        return self.values[..., nocc-1].max()

    @property
    def cbb(self):
        """Conduction bands minimum (bottom) value."""
        nocc = self.nocc
        if nocc == self.values.shape[-1]:
            raise BandsMissingException("No conduction bands")
        return self.values[..., nocc].min()

    @property
    def gap(self):
        """The band gap."""
        return self.cbb - self.vbt

    def stick_fermi(self, value, epsilon=1e-12):
        """
        Shifts the the Fermi level.
        Args:
            value (str): the new Fermi level position: one of
            'midgap', 'cbb', 'vbt';
            epsilon (float): infinitesimal term to separate
            the Fermi level and bands in case `value`='cbb'
            or 'vbt';
        """
        self.fermi = dict(
            midgap=.5 * (self.cbb + self.vbt),
            cbb=self.cbb - epsilon,
            vbt=self.vbt + epsilon,
        )[value]

    def canonize_fermi(self):
        """
        Places the Fermi level in the middle of the band gap.
        Shifts the energy scale zero to the Fermi level.
        """
        try:
            self.stick_fermi("midgap")
        except FermiCrossingException:
            pass
        self.values -= self.fermi
        self.fermi = 0


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
