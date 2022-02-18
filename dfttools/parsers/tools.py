"""
Parsing Local JSON file format.
"""
import importlib

from .generic import AbstractJSONParser, IdentifiableParser, ParseError
from ..util import loads
from ..simple import unit_cell, band_structure

valid_containers = ("dfttools.types.Basis", "dfttools.types.Cell", "dfttools.types.Grid",
                    "dfttools.types.RealSpaceBasis", "dfttools.types.ReciprocalSpaceBasis",
                    "dfttools.types.CrystalCell", "dfttools.types.CrystalGrid", "dfttools.types.BandsPath",
                    "dfttools.types.BandsGrid")
valid_containers_uc = ("dfttools.types.CrystalCell",)
valid_containers_bands = ("dfttools.types.BandsPath", "dfttools.types.BandsGrid")

lookup_container = {}
lookup_type_string = {}

for i in valid_containers:
    _package, _module, _class = i.split(".")
    assert _package == "dfttools"
    module = importlib.import_module(".".join((_package, _module)))
    lookup_container[i] = getattr(module, _class)
    lookup_type_string[lookup_container[i]] = i


class JSONStorage(AbstractJSONParser, IdentifiableParser):
    """
    Handles parsing of all json-serialized data from this package.
    """

    @staticmethod
    def valid_header(header):
        return "\"type\"" in header and "\"dfttools." in header

    loads = staticmethod(loads)

    @staticmethod
    def __pick_class__(o):
        if "type" not in o:
            raise ParseError("Not a dfttools storage")
        t = o["type"]
        if t not in lookup_container:
            raise ParseError("Unknown type: {}".format(t))
        return lookup_container[t]

    @staticmethod
    def __common_class__(objs):
        if not isinstance(objs, list):
            objs = [objs]
        c = set(i.__class__ for i in objs)
        if len(c) > 1:
            raise ParseError("Several different classes found: {}".format(c))
        return c.pop()

    def assemble(self, index=None):
        """
        Assembles a python object from this container.
        Args:
            index (int): the index of the object, if available;

        Returns:
            One of the objects defined in `dfttools.types`, `dfttools.utypes`.
        """
        data = self.json
        if isinstance(self.json, list):
            if index is None:
                return list(self.__pick_class__(i).from_state_dict(i) for i in self.json)
            data = self.json[index]
        return self.__pick_class__(data).from_state_dict(data)

    @unit_cell
    def cells(self, index=None):
        """
        Retrieves the atomic structure data.
        Args:
            index (int): the index of the structure, if available;

        Returns:
            The unit cell.
        """
        result = self.assemble(index=index)
        common_class = self.__common_class__(result)
        if lookup_type_string[common_class] not in valid_containers_uc:
            raise ParseError("The container is not an atomic structure: {}".format(common_class))
        return result

    @band_structure
    def bands(self, index=-1):
        """
        Retrieves the band structure data.
        Args:
            index (int): the index of the band structure, if available;

        Returns:
            Band structure.
        """
        result = self.assemble(index=index)
        common_class = self.__common_class__(result)
        if lookup_type_string[common_class] not in valid_containers_bands:
            raise ParseError("The container is not bands: {}".format(common_class))
        return result


jsons = JSONStorage
