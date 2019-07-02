"""
Parsing Local JSON file format.
"""
import importlib

from .generic import AbstractJSONParser, ParseError
from ..util import loads
from ..simple import unit_cell, band_structure

valid_containers = ("dfttools.types.Basis", "dfttools.types.UnitCell", "dfttools.types.Grid",
                    "dfttools.utypes.RealSpaceBasis", "dfttools.utypes.ReciprocalSpaceBasis",
                    "dfttools.utypes.CrystalCell", "dfttools.utypes.CrystalGrid", "dfttools.utypes.BandsPath",
                    "dfttools.utypes.BandsGrid")
valid_containers_uc = ("dfttools.utypes.CrystalCell",)
valid_containers_bands = ("dfttools.utypes.BandsPath", "dfttools.utypes.BandsGrid")

lookup_container = {}
lookup_type_string = {}

for i in valid_containers:
    _package, _module, _class = i.split(".")
    assert _package == "dfttools"
    module = importlib.import_module(".".join((_package, _module)))
    lookup_container[i] = getattr(module, _class)
    lookup_type_string[lookup_container[i]] = i


class JSONStorage(AbstractJSONParser):
    """
    Handles parsing of all json-serialized data from this package.
    """

    @staticmethod
    def valid_header(header):
        return "\"type\"" in header and "\"dfttools." in header

    loads = staticmethod(loads)

    def __pick_class__(self):
        if "type" not in self.json:
            raise ParseError("Not a dfttools storage")
        t = self.json["type"]
        if t not in lookup_container:
            raise ParseError("Unknown type: {}".format(t))
        return lookup_container[t]

    def assemble(self):
        """
        Assembles a python object from this container.

        Returns:
            One of the objects defined in `dfttools.types`, `dfttools.utypes`.
        """
        return self.__pick_class__().from_json(self.json)

    @unit_cell
    def unitCell(self):
        """
        Retrieves the atomic structure data.

        Returns:
            The unit cell.
        """
        result = self.assemble()
        if lookup_type_string[result.__class__] not in valid_containers_uc:
            raise ParseError("The container is not an atomic structure: {}".format(result.__class__))
        return result

    @band_structure
    def bands(self):
        """
        Retrieves the band structure data.

        Returns:
            Band structure.
        """
        result = self.assemble()
        if lookup_type_string[result.__class__] not in valid_containers_bands:
            raise ParseError("The container is not bands: {}".format(result.__class__))
        return result


jsons = JSONStorage
