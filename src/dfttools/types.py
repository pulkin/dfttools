"""
This submodule enhances `pycoordinates` with units-aware arrays.
"""
import numpy

from pycoordinates import Basis, Cell, Grid
from pycoordinates.util import roarray_copy
from functools import cached_property
from attr import attrs, attrib
from . import util


element_type = numpy.array("Ca").dtype


class UnitsMixin:
    """
    A core mixin to work with units.
    """
    default_units = {}

    def __attrs_post_init__(self):
        for k, v in self.default_units.items():
            if k in dir(self):
                target = getattr(self, k)
                object.__setattr__(self, k, util.array(target, units=v))

    @cached_property
    def vectors_inv(self):
        result = super().vectors_inv
        units = getattr(self.vectors, "units", None)
        return util.ArrayWithUnits(result, units=None if units is None else f"1/({units})")


@attrs(frozen=True, eq=False)
class RealSpaceBasis(UnitsMixin, Basis):
    """
    Basis in real space.
    """

    default_units = dict(vectors="angstrom")


@attrs(frozen=True, eq=False)
class ReciprocalSpaceBasis(UnitsMixin, Basis):
    """
    Basis in reciprocal space.
    """

    default_units = dict(vectors="1/angstrom")


@attrs(frozen=True, eq=False)
class CrystalCell(UnitsMixin, Cell):
    """
    A unit cell of a crystal.
    """

    default_units = dict(vectors="angstrom", values=None)


@attrs(frozen=True, eq=False)
class CrystalGrid(UnitsMixin, Grid):
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


def convert_fermi(fermi) -> util.ArrayWithUnits:
    if fermi is None:
        return
    else:
        return roarray_copy(fermi)


def check_fermi(instance, attribute: str, fermi: util.ArrayWithUnits):
    if fermi is not None and fermi.shape != ():
        raise ValueError(f"fermi.shape={fermi.shape} is not a scalar")


@attrs(frozen=True, eq=False)
class FermiMixin:
    """
    A mixin to add the Fermi attribute.

    Args:
        fermi (float): the Fermi level value;
    """
    fermi = attrib(converter=convert_fermi, validator=check_fermi, default=None, kw_only=True)

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

    def compute_fermi_level(self, value, epsilon=1e-12):
        """
        Computes the Fermi level.
        Args:
            value (str): the Fermi level position: one of
            'midgap', 'cbb', 'vbt';
            epsilon (float): infinitesimal term to separate
            the Fermi level and bands in case `value`='cbb'
            or 'vbt';
        """
        return {
            "midgap": .5 * (self.cbb + self.vbt),
            "cbb": self.cbb - epsilon,
            "vbt": self.vbt + epsilon,
        }[value]

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
        return self.copy(fermi=self.compute_fermi_level(value, epsilon))

    def canonize_fermi(self):
        """
        Places the Fermi level in the middle of the band gap.
        Shifts the energy scale zero to the Fermi level.
        """
        try:
            fermi = self.compute_fermi_level("midgap")
        except FermiCrossingException:
            fermi = self.fermi
        return self.copy(fermi=0, values=self.values-fermi)


@attrs(frozen=True, eq=False)
class BandsPath(FermiMixin, UnitsMixin, Cell):
    """
    A class describing a band structure along a path.
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
    as_grid.__doc__ = Cell.as_grid.__doc__

    def interpolate(self, *args, **kwargs):
        result = super(BandsPath, self).interpolate(*args, **kwargs)
        return BandsPath(result.vectors, result.coordinates, result.values, meta=result.meta, fermi=self.fermi)
    interpolate.__doc__ = Cell.interpolate.__doc__


@attrs(frozen=True, eq=False)
class BandsGrid(FermiMixin, UnitsMixin, Grid):
    """
     A class describing a band structure on a grid.
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
    as_cell.__doc__ = Grid.as_cell.__doc__

    def interpolate_to_cell(self, *args, **kwargs):
        result = super(BandsGrid, self).interpolate_to_cell(*args, **kwargs)
        return BandsPath(result.vectors, result.coordinates, result.values, meta=result.meta, fermi=self.fermi)
    interpolate_to_cell.__doc__ = Grid.interpolate_to_cell.__doc__

    def interpolate_to_path(self, *args, **kwargs):
        result = super(BandsGrid, self).interpolate_to_path(*args, **kwargs)
        return BandsPath(result.vectors, result.coordinates, result.values, meta=result.meta, fermi=self.fermi)
    interpolate_to_path.__doc__ = Grid.interpolate_to_path.__doc__

    def interpolate_to_grid(self, *args, **kwargs):
        result = super(BandsGrid, self).interpolate_to_grid(*args, **kwargs)
        return BandsGrid(result.vectors, result.coordinates, result.values, meta=result.meta, fermi=self.fermi)
    interpolate_to_grid.__doc__ = Grid.interpolate_to_grid.__doc__
