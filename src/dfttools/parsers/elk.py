"""
Parsing `ELK <http://elk.sourceforge.net/>`_ files.
"""
import math
import re

import numpy
import numericalunits

from .generic import parse, cre_nonspace, cre_float, cre_word, AbstractParser
from ..simple import band_structure, unit_cell
from ..types import UnitCell, Basis
from . import default_real_space_basis

class UnitCellsParser(AbstractParser):
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
        kw = re.compile(r'\b'+section+r'\b')
        self.parser.skip(kw)
        self.parser.nextLine()

    def __next_unit_cell__(self):
        
        self.parser.save()
        try:
            self.__tosection__('scale')
            scale = [self.parser.nextFloat()*numericalunits.aBohr]*3
            
        except StopIteration:
            scale = [numericalunits.aBohr]*3
            
        self.parser.pop()
        
        for i in range(3):
            self.parser.save()
            try:
                self.__tosection__('scale{:d}'.format(i+1))
                scale[i] = self.parser.nextFloat()*numericalunits.aBohr
            except StopIteration:
                pass
            self.parser.pop()

        self.parser.save()
        self.__tosection__('avec')
        vectors = self.parser.nextFloat(n = (3,3)) * numpy.array(scale)[:,numpy.newaxis]
        self.parser.pop()
        
        self.__tosection__('atoms')
        n_sp = self.parser.nextInt()
        self.parser.nextLine()
        
        coordinates = []
        values = []
        
        for s in range(n_sp):
            
            self.parser.skip('\'')
            name = self.parser.nextMatch(cre_word)
            self.parser.nextLine()
            
            n_at = self.parser.nextInt()
            self.parser.nextLine()
            
            coordinates.append(self.parser.nextFloat(n = (n_at, 6))[:,:3])
            values = values + [name]*n_at
            
        return UnitCell(
            default_real_space_basis(vectors),
            numpy.concatenate(coordinates, axis = 0),
            values,
        )
    
    @unit_cell
    def unitCells(self):
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
    
class Input(UnitCellsParser):
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
    
    # Non need to place @unit_cell here: inherits UnitCellsParser.unitCells()
    def unitCell(self):
        """
        Retrieves the unit cell specified in the input file.
        
        Returns:
        
            The unit cell with atomic positions.
        """
        self.parser.reset()
        return self.__next_unit_cell__()
            
    def kp_path(self, basis = None):
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
            basis = self.__next_unit_cell__().reciprocal()
        
        # Read data from input
        self.parser.skip('plot1d\n')
        nodes = self.parser.nextInt()
        points = self.parser.nextInt()
        self.parser.nextLine()
        nodes = self.parser.nextFloat(n = (nodes, 3))
        
        # Calculate path lengths
        nodes_c = basis.transform_to_cartesian(nodes)
        path_dst = numpy.cumsum(
            (
                (nodes_c[1:] - nodes_c[:-1])**2
            ).sum(axis = 1)
        )
        
        # Calculate positions of kpoints on the path
        kp_on_path = numpy.linspace(0,path_dst[-1],points)
        
        # Calculate coordinates of kpoints
        coordinates = []
        for d in kp_on_path:
            
            # Determine the segment j
            j = numpy.searchsorted(path_dst, d)
            
            fraction = (path_dst[j] - d)/(path_dst[j] - (path_dst[j-1] if j>0 else 0))
            coordinates.append(nodes[j,:]*fraction + nodes[j+1,:]*(1-fraction))
            
        return numpy.array(coordinates)
    
class Output(AbstractParser):
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
    def unitCell(self):
        """
        Retrieves the unit cell.
        
        Returns:
        
            A Cell object with atomic coordinates.
        """
        self.parser.reset()
        
        self.parser.skip('Lattice vectors :')
        
        vecs = self.parser.nextFloat(n = (3,3))*numericalunits.aBohr
        
        n = self.parser.intAfter('Total number of atoms per unit cell :')
        n_read = 0
        
        coordinates = []
        values = []
        
        while n_read<n:
            
            self.parser.skip('Species :')
            self.parser.nextInt()
            name = self.parser.nextMatch(cre_nonspace)[1:-1]
            
            self.parser.skip('atomic positions (lattice)')
            self.parser.nextLine()
            coords = self.parser.nextFloat(n = '\n \n').reshape((-1,7))[:,(1,2,3)]
            
            n_read += coords.shape[0]
            coordinates.append(coords)
            values += [name]*coords.shape[0]
            
        coordinates = numpy.concatenate(coordinates, axis = 0)
        return UnitCell(
            default_real_space_basis(vecs),
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
        
        vecs = self.parser.nextFloat(n = (3,3))*2*math.pi/numericalunits.aBohr
        return Basis(vecs)
        
class Bands(AbstractParser):
    """
    Class for parsing band structure from BAND.OUT file.
    
    Args:
    
        data (string): contents of Elk BAND.OUT file
    """
    
    @staticmethod
    def valid_filename(name):
        return name == "BAND.OUT"
    
    @band_structure
    def bands(self):
        """
        Retrieves the band structure and strores it into a flattened UnitCell.
        
        Returns:
        
            A Unit cell with the band structure.
        """
        self.parser.reset()
        
        values = []
        while self.parser.present(cre_float):
            a = self.parser.nextFloat(n = '\n     \n')
            self.parser.nextLine(2)
            values.append(a[1::2]*numericalunits.Hartree)
            coordinates = a[::2]

        return UnitCell(
            Basis((1,)),
            coordinates[:,numpy.newaxis],
            numpy.array(values).swapaxes(0,1),
        )

# Lower case versions
input = Input
bands = Bands
output = Output    
unitcells = UnitCellsParser
