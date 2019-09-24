"""
Parsing `wannier90 <http://www.wannier.org/>`_ files.
"""
import numericalunits

from .generic import cre_var_name, cre_word, cre_non_space, re_int, cre_int, cre_float, AbstractTextParser, \
    IdentifiableParser
from ..simple import unit_cell
from ..utypes import CrystalCell


class Input(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing parameter values from wannier90 input files.

    Args:
        data (str): contents of the wannier90 input file
    """
    @staticmethod
    def valid_header(header):
        l = header.lower()
        return "begin atoms_frac" in l and "begin unit_cell_cart" in l

    @staticmethod
    def valid_filename(name):
        return name.endswith(".win")

    @unit_cell
    def unitCell(self):
        """
        Retrieves atomic position data.
        Returns:
            The unit cell.
        """

        self.parser.reset()
        vectors = self.parser.float_after("begin unit_cell_cart", (3, 3)) * numericalunits.angstrom
        coordinates_and_atoms = self.parser.match_after(
            "begin atoms_frac", cre_non_space, "end atoms_frac").reshape(-1, 4)
        coordinates = coordinates_and_atoms[:, 1:].astype(float)
        values = coordinates_and_atoms[:, 0]
        return CrystalCell(
            vectors,
            coordinates,
            values,
        )


# Lower case versions
input = Input
