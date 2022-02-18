"""
Parsing `ELK <http://elk.sourceforge.net/>`_ files.
"""
import math
import re

import numericalunits
import numpy

from .generic import cre_non_space, cre_float, cre_word, AbstractTextParser, IdentifiableParser
from ..simple import band_structure, unit_cell
from ..types import Basis, CrystalCell, BandsPath
from ..util import array


class CellsParser(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing elk GEOMETRY_OPT.OUT.
    
    Args:
    
        data (string): contents of GEOMETRY_OPT.OUT file
    """

    @staticmethod
    def valid_header(header):
        return "! Lattice and atomic position optimisation steps" in header

    @staticmethod
    def valid_filename(name):
        return name == "GEOMETRY_OPT.OUT"

    def __tosection__(self, section):
        kw = re.compile(r'\b' + section + r'\b')
        self.parser.skip(kw)
        self.parser.next_line()

    def __next_unit_cell__(self):

        self.parser.save()
        try:
            self.__tosection__('scale')
            scale = [self.parser.next_float() * numericalunits.aBohr] * 3

        except StopIteration:
            scale = [numericalunits.aBohr] * 3

        self.parser.pop()

        for i in range(3):
            self.parser.save()
            try:
                self.__tosection__('scale{:d}'.format(i + 1))
                scale[i] = self.parser.next_float() * numericalunits.aBohr
            except StopIteration:
                pass
            self.parser.pop()

        self.parser.save()
        self.__tosection__('avec')
        vectors = self.parser.next_float(n=(3, 3)) * numpy.array(scale)[:, numpy.newaxis]
        self.parser.pop()

        self.__tosection__('atoms')
        n_sp = self.parser.next_int()
        self.parser.next_line()

        coordinates = []
        values = []

        for s in range(n_sp):
            self.parser.skip('\'')
            name = self.parser.next_match(cre_word)
            self.parser.next_line()

            n_at = self.parser.next_int()
            self.parser.next_line()

            coordinates.append(self.parser.next_float(n=(n_at, 6))[:, :3])
            values = values + [name] * n_at

        return CrystalCell(
            vectors,
            numpy.concatenate(coordinates, axis=0),
            values,
        )

    @unit_cell
    def cells(self):
        """
        Retrives the geometry optimization steps as unit cells.
        
        Returns:
        
            An array with unit cells.
        """

        self.parser.reset()
        result = []

        while True:
            try:
                result.append(self.__next_unit_cell__())
            except StopIteration:
                break

        return result


class Input(CellsParser, IdentifiableParser):
    """
    Class for parsing elk.in input file.
    
    Args:
    
        data (string): contents of elk.in file
    """

    @staticmethod
    def valid_filename(name):
        return name == "elk.in"

    @staticmethod
    def valid_header(header):
        l = header.lower()
        return "avec" in l and "atoms" in l and "ngridk" in l

    # inherits @unit_cell
    def cell(self):
        """
        Retrieves the unit cell specified in the input file.
        
        Returns:
        
            The unit cell with atomic positions.
        """
        self.parser.reset()
        return self.__next_unit_cell__()

    def kp_path(self, basis=None):
        """
        Calculates k-point path from input and returns it as an array of
        crystal coordinates.
        
        Kwargs:
        
            basis (Basis): a reciprocal basis for the k-vectors.
            
        Returns:
        
            An array of k-point coordinates in reciprocal basis.
        """
        self.parser.reset()
        if basis is None:
            basis = self.__next_unit_cell__().reciprocal

        # Read data from input
        self.parser.skip('plot1d\n')
        nodes = self.parser.next_int()
        points = self.parser.next_int()
        self.parser.next_line()
        nodes = self.parser.next_float(n=(nodes, 3))

        # Calculate path lengths
        nodes_c = basis.transform_to_cartesian(nodes)
        path_dst = numpy.cumsum(
            (
                    (nodes_c[1:] - nodes_c[:-1]) ** 2
            ).sum(axis=1)
        )

        # Calculate positions of kpoints on the path
        kp_on_path = numpy.linspace(0, path_dst[-1], points)

        # Calculate coordinates of kpoints
        coordinates = []
        for d in kp_on_path:
            # Determine the segment j
            j = numpy.searchsorted(path_dst, d)

            fraction = (path_dst[j] - d) / (path_dst[j] - (path_dst[j - 1] if j > 0 else 0))
            coordinates.append(nodes[j, :] * fraction + nodes[j + 1, :] * (1 - fraction))

        return numpy.array(coordinates)


class Output(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing INFO.OUT of elk output.
    
    Args:
    
        data (string): contents of INFO.OUT file
    """

    @staticmethod
    def valid_filename(name):
        return name == "INFO.OUT"

    @staticmethod
    def valid_header(header):
        return "+----------------------------+" in header and "Elk version" in header

    @unit_cell
    def cell(self):
        """
        Retrieves the unit cell.
        
        Returns:
        
            A Cell object with atomic coordinates.
        """
        self.parser.reset()

        self.parser.skip('Lattice vectors :')

        vecs = self.parser.next_float(n=(3, 3)) * numericalunits.aBohr

        n = self.parser.int_after('Total number of atoms per unit cell :')
        n_read = 0

        coordinates = []
        values = []

        while n_read < n:
            self.parser.skip('Species :')
            self.parser.next_int()
            name = self.parser.next_match(cre_non_space)[1:-1]

            self.parser.skip('atomic positions (lattice)')
            self.parser.next_line()
            coords = self.parser.next_float(n='\n \n').reshape((-1, 7))[:, (1, 2, 3)]

            n_read += coords.shape[0]
            coordinates.append(coords)
            values += [name] * coords.shape[0]

        coordinates = numpy.concatenate(coordinates, axis=0)
        return CrystalCell(
            vecs,
            coordinates,
            values,
        )

    def reciprocal(self):
        """
        Retrieves the reciprocal basis.
        
        Returns:
        
            A reciprocal basis.
        """
        self.parser.reset()

        self.parser.skip('Reciprocal lattice vectors :')

        vecs = self.parser.next_float(n=(3, 3)) * 2 * math.pi / numericalunits.aBohr
        return Basis(array(vecs, units="1/angstrom"))


class Bands(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing band structure from BAND.OUT file.
    
    Args:
    
        data (string): contents of Elk BAND.OUT file
    """

    @staticmethod
    def valid_filename(name):
        return name == "BAND.OUT"

    @staticmethod
    def valid_header(header):
        raise NotImplementedError

    @band_structure
    def bands(self):
        """
        Retrieves the band structure and stores it into a flattened Cell.
        
        Returns:
        
            A Unit cell with the band structure.
        """
        self.parser.reset()

        values = []
        while self.parser.present(cre_float):
            a = self.parser.next_float(n='\n     \n')
            self.parser.next_line(2)
            values.append(a[1::2] * numericalunits.Hartree)
            coordinates = a[::2]

        return BandsPath(
            [(1,)],
            coordinates[:, numpy.newaxis],
            numpy.array(values).swapaxes(0, 1),
        )


# Lower case versions
input = Input
bands = Bands
output = Output
cells = CellsParser
