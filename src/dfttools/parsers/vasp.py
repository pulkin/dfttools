"""
Parsing `VASP <https://www.vasp.at/>`_ files.
"""
import numericalunits

from .generic import AbstractParser
from ..simple import band_structure, unit_cell
from ..types import Basis, UnitCell

class OutputParser(AbstractParser):
    """
    Class for parsing VASP OUTCAR.
    
    Args:
    
        data (string): contents of OUTCAR file
    """
        
    @staticmethod
    def valid_header(header):
        return header.startswith(" vasp.5.4")
        
    @staticmethod
    def valid_filename(name):
        return name == "OUTCAR"
        
    def fermi(self):
        """
        Retrieves Fermi energies.
        
        Returns:
        
            A numpy array containing Fermi energies for each MD step.
        """
        self.parser.reset()
        result = []
        while self.parser.present("E-fermi :"):
            self.parser.skip("E-fermi :")
            result.append(self.parser.nextFloat()*numericalunits.eV)
        return result
    
    def __reciprocal__(self):
        self.parser.skip("reciprocal lattice vectors")
        self.parser.nextLine()
        return Basis(
            self.parser.nextFloat((3,6))[:,3:],
        )
                
    def __kpoints__(self):
        self.parser.skip("k-points in reciprocal lattice and weights: K-points")
        return self.parser.nextFloat("\n \n").reshape(-1,4)[:,:3]
        
    @band_structure
    def bands(self):
        """
        Retrieves bands.
            
        Returns:
        
            A UnitCells with the band structure.
        """
        self.parser.reset()
        basis = self.__reciprocal__()
        k = self.__kpoints__()
        
        self.parser.skip("E-fermi :")
        fermi = self.parser.nextFloat()*numericalunits.eV
        
        e = []
        for i in range(len(k)):
            self.parser.skip("k-point")
            self.parser.skip(str(i+1))
            self.parser.nextLine(2)
            e.append(self.parser.nextFloat("\n\n").reshape(-1,3)[:,1]*numericalunits.eV)
            
        result = UnitCell(
            basis,
            k,
            e,
        )
        result.meta["Fermi"] = fermi
        return result
