"""
Parsing `VASP <https://www.vasp.at/>`_ files.
"""
import numericalunits

from .generic import AbstractTextParser, IdentifiableParser
from ..simple import band_structure
from ..types import CrystalCell, BandsPath


class Output(AbstractTextParser, IdentifiableParser):
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
            result.append(self.parser.next_float() * numericalunits.eV)
        return result

    def __reciprocal__(self):
        self.parser.skip("reciprocal lattice vectors")
        self.parser.next_line()
        return self.parser.next_float((3, 6))[:, 3:] / numericalunits.angstrom

    def __kpoints__(self):
        self.parser.skip("k-points in reciprocal lattice and weights: K-points")
        return self.parser.next_float("\n \n").reshape(-1, 4)[:, :3]

    @band_structure
    def bands(self):
        """
        Retrieves bands.
            
        Returns:
        
            Cell with the band structure.
        """
        self.parser.reset()
        basis = self.__reciprocal__()
        k = self.__kpoints__()

        self.parser.skip("E-fermi :")
        fermi = self.parser.next_float() * numericalunits.eV

        e = []
        for i in range(len(k)):
            self.parser.skip("k-point")
            self.parser.skip(str(i + 1))
            self.parser.next_line(2)
            e.append(self.parser.next_float("\n\n").reshape(-1, 3)[:, 1] * numericalunits.eV)

        result = BandsPath(
            basis,
            k,
            e,
            fermi=fermi,
        )
        return result


class Structure(AbstractTextParser):
    """
    Class for parsing VASP POSCAR.
    
    Args:
    
        data (string): contents of POSCAR file
    """

    def cell(self, names):
        self.parser.reset()
        self.parser.next_line()
        scale = self.parser.next_float()
        self.parser.next_line()
        vectors = self.parser.next_float((3, 3))
        if scale > 0:
            scale *= numericalunits.angstrom
        else:
            raise NotImplemented
        vectors *= scale
        self.parser.next_line(2)
        nat = self.parser.next_int("\n")
        self.parser.next_line()
        l = self.parser.next_line().lower()
        if l[0] == 'c' or l[0] == 'k':
            c_basis = 'cartesian'
        else:
            c_basis = None
        coords = self.parser.next_float((sum(nat), 3))
        atoms = []
        for name, number in zip(names, nat):
            atoms += [name] * number
        return CrystalCell(
            vectors,
            coords if c_basis is None else coords * scale,
            atoms,
            c_basis=c_basis,
        )
