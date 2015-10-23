"""
Parsing `OpenMX <http://openmx-square.org/>`_ files.
"""
import re
import warnings
import math
import json
import os
import os.path

import numpy
import numericalunits

from .generic import parse, cre_varName, cre_word, cre_nonspace, re_int, cre_int, cre_float, AbstractParser, ParseError
from .structure import cube
from ..simple import band_structure, unit_cell, guess_parser, parse, tag_method
from ..types import UnitCell, Basis

def populations(s):
    """
    Parses JSON with Lowdin populations. Replaces corresponding arrays
    with numpy objects. Adds units where needed.
    
    Args:
    
        s (str): file contents.
    
    Returns:
    
        A dict with JSON data.
    """
    result = json.loads(s)
    for field in ("k", "bands", "energies", "weights"):
        result[field] = numpy.array(result[field])
        
    result["energies"] *= 2*numericalunits.Ry
        
    for field in result["basis"].keys():
        result["basis"][field] = numpy.array(result["basis"][field])
        
    return result
    
def joint_populations(files):
    """
    Collects several files with Lowdin populations and parses them into
    a single object.
    
    Args:
    
        files (list of str): file contents in an array.
    
    Returns:
    
        A dict with joint JSON data.
    """
    if len(files) == 0:
        raise ValueError("Empty array passed")
        
    parsed = {}
    for f in files:
        p = populations(f)
        parsed[p["k-id"]] = p

    collected = {
        "k": [],
        "energies": [],
        "weights": [],
    }
    for i in range(len(parsed)):
        
        if not i in parsed:
            raise ValueError("Missing data at k-point #{:d}".format(i))
        p = parsed[i]
        p2 = parsed[0]
        
        if i>0:
            if not numpy.all(p["bands"] == p2["bands"]):
                raise ValueError("Different bands reported at k #0 and k #{:d}".format(i))
            for k in p["basis"].keys() + p2["basis"].keys():
                if not k in p["basis"] or not k in p2["basis"]:
                    raise ValueError("The basis description '{}' is missing for k #0 or k #{:d}".format(k,i))
                if not numpy.all(p["basis"][k] == p2["basis"][k]):
                    raise ValueError("The basis description '{}' is different for k #0 and k #{:d}".format(k,i))
        
        for c in collected.keys():
            collected[c].append(p[c])
        
    result = parsed[0]
    del result["k-id"]
    for c in collected.keys():
        result[c] = numpy.array(collected[c])
        
    return result
    
def dos(s):
    """
    Parses JSON with DoS. Replaces corresponding arrays with numpy
    objects. Adds units where needed.
    
    Args:
    
        s (str): file contents.
    
    Returns:
    
        A dict with JSON data.
    """
    result = json.loads(s)
    for field in ("energy", "DOS"):
        result[field] = numpy.array(result[field])
        
    result["energy"] *= 2*numericalunits.Ry
        
    for field in result["basis"].keys():
        result["basis"][field] = numpy.array(result["basis"][field])
        
    return result
    
class Input(AbstractParser):
    """
    Class for parsing parameter values from OpenMX input files.
    
    Args:
    
        data (str): contents of OpenMX input file
    """
    
    @staticmethod
    def valid_header(header):
        l = header.lower()
        return "definition.of.atomic.species" in l
        
    def systemName(self):
        return self.getNonSpaced("system.name")
        
    def getWord(self, parameter):
        """
        A shortcut to parser.StringParser.matchAfter designed to obtain
        word-like parameter values from textual configuration files.
        
        Args:
        
            parameter (str): parameter name
            
        Returns:
        
            parameter value
        """
        self.parser.reset()
        return self.parser.matchAfter(parameter,cre_word)
        
    def getNonSpaced(self, parameter):
        """
        A shortcut to parser.StringParser.matchAfter designed to obtain
        parameter values without spaces from textual configuration files.
        
        Args:
        
            parameter (str): parameter name
            
        Returns:
        
            parameter value
        """
        self.parser.reset()
        return self.parser.matchAfter(parameter,cre_nonspace)

    def getFloat(self, parameter):
        """
        A shortcut to parser.StringParser.matchAfter designed to obtain
        float parameter values from textual configuration files.
        
        Args:
        
            parameter (str): parameter name
            
        Returns:
        
            parameter value
        """
        self.parser.reset()
        return self.parser.floatAfter(parameter)

    def getInt(self, parameter):
        """
        A shortcut to parser.StringParser.matchAfter designed to obtain
        integer parameter values from textual configuration files.
        
        Args:
        
            parameter (str): parameter name
            
        Returns:
        
            parameter value
        """
        self.parser.reset()
        return self.parser.intAfter(parameter)

    @unit_cell
    def unitCell(self, l = None, r = None):
        """
        Retrieves atomic position data.
        
        Kwargs:
        
            l,r (qetools.cell.Cell): left lead and right lead cells.
            This information is required for parsing the cell from NEGF
            calculation input file.
        
        Returns:
        
            A cell with atom coordinates in **m**. In ``Cell.values``
            chemical captions are stored.
            
        Raises:
        
            ValueError: left and right lead cells are not specified for
            NEGF input file.
        """
       
        self.parser.reset()

        if not self.parser.present("<Atoms.UnitVectors"):
            if l is None or r is None:
                raise ValueError("The input file does not specify unit cell dimensions (NEGF calculation) but the left and right leads are not specified")
            no_vectors = True
            
        else:
            no_vectors = False
            
        n = self.getInt("Atoms.Number")
        units = self.getWord("Atoms.SpeciesAndCoordinates.Unit")

        self.parser.save()
        self.parser.skip("<Atoms.SpeciesAndCoordinates")
        coordinates = numpy.zeros((n,3))
        values = []
        for i in range(n):
            self.parser.nextInt()
            values.append(self.parser.nextMatch(cre_word))
            coordinates[i,:] = self.parser.nextFloat(3)
            self.parser.nextLine()
        self.parser.pop()
        
        if units.lower() == "ang":
            coordinates *= numericalunits.angstrom
        elif units.lower() == "au":
            coordinates *= numericalunits.aBohr
            
        if no_vectors:
            
            nl = self.parser.intAfter("LeftLeadAtoms.Number")
            nr = self.parser.intAfter("RightLeadAtoms.Number")

            self.parser.skip("<LeftLeadAtoms.SpeciesAndCoordinates")
            coordinatesl = numpy.zeros((nl,3))
            valuesl = []
            for i in range(nl):
                self.parser.nextInt()
                valuesl.append(self.parser.nextMatch(cre_word))
                coordinatesl[i,:] = self.parser.nextFloat(3)
                self.parser.nextLine()
            self.parser.reset()
    
            self.parser.skip("<RightLeadAtoms.SpeciesAndCoordinates")
            coordinatesr = numpy.zeros((nr,3))
            valuesr = []
            for i in range(nr):
                self.parser.nextInt()
                valuesr.append(self.parser.nextMatch(cre_word))
                coordinatesr[i,:] = self.parser.nextFloat(3)
                self.parser.nextLine()
            self.parser.reset()
                
            if units.lower() == "ang":
                coordinatesl *= numericalunits.angstrom
                coordinatesr *= numericalunits.angstrom
            elif units.lower() == "au":
                coordinatesl *= numericalunits.aBohr
                coordinatesr *= numericalunits.aBohr

            dl = (l.cartesian()-coordinatesl).sum(axis = 0)/l.size()
            dr = (r.cartesian()-coordinatesr).sum(axis = 0)/r.size()
            shape = l.vectors.copy()
            shape[0,:] = dl-dr-l.vectors[0,:]
            coordinates = coordinates + dl - l.vectors[0,:]
            
        else:
            
            units_cell = self.parser.matchAfter("Atoms.UnitVectors.Unit",cre_word).lower()
            shape = self.parser.floatAfter("<Atoms.UnitVectors",n = (3,3))
    
            if units_cell == "ang":
                shape *= numericalunits.angstrom
            elif units_cell == "au":
                shape *= numericalunits.aBohr

        return UnitCell(
            Basis(shape),
            coordinates,
            values,
            c_basis = None if units.lower() == "frac" else "cartesian"
        )

class Output(AbstractParser):
    """
    Class for parsing parameter values from OpenMX output files.
    
    Args:
    
        data (string): contents of OpenMX output file
    """
        
    @staticmethod
    def valid_header(header):
        return "Welcome to OpenMX" in header and "T. Ozaki" in header
    
    def version(self):
        """
        Retrieves OpenMX version as reported in the output.
        
        Returns:
        
            OpenMX program version as string.
        """
        self.parser.reset()
        self.parser.skip(re.compile("Welcome to OpenMX\s+Ver\."))
        return self.parser.nextMatch(cre_varName,n = "\n")[0]
        
    def total(self):
        """
        Retrieves total energy calculated.
        
        Returns:
        
            An array of floats with total energy per each SCF cycle.
        """
        self.parser.reset()
        result = []
        
        while self.parser.present("Utot  ="):
            self.parser.skip("Utot  =")
            result.append(self.parser.nextFloat()*2*numericalunits.Ry)
            
        return numpy.array(result)
        
    def nat(self):
        """
        Retrieves number of atoms.
        
        Returns:
        
            Number of atoms for the first relaxation step.
        """
        self.parser.reset()
        self.parser.skip("MD or geometry opt. at MD")
        self.parser.skip("maximum force")
        
        n = 0
        while self.parser.closest(("XYZ(ang)","***")) == 0:
            self.parser.skip("XYZ(ang)")
            n += 1
            
        return n
        
    def unitCells(self, startingCell, noraise = False):
        """
        Retrieves atomic positions data for relax calculation.
        
        Args:
        
            startingCell (qetools.cell.Cell): a unit cell from the input
            file. It is required since no chemical captions are written
            in the output.
            
        Kwargs:
        
            noraise (bool): retirieves as much structures as possible
            without raising exceptions.
        
        Returns:
        
            A set of all unit cells found.
        """
        self.parser.reset()
        cells = []
        
        while self.parser.present("lattice vectors (bohr)"):
            
            try:
                
                self.parser.skip("lattice vectors (bohr)")
                shape = self.parser.nextFloat((3,3))*numericalunits.aBohr
                    
                self.parser.skip("MD or geometry opt. at MD")
                self.parser.skip("maximum force")
                coordinates = []
                
                while self.parser.closest(("XYZ(ang)","***")) == 0:
                    
                    self.parser.skip("XYZ(ang)")
                    coordinates.append(self.parser.nextFloat(3)*numericalunits.angstrom)
                    
                cells.append(UnitCell(
                    Basis(shape),
                    coordinates,
                    startingCell.values,
                    c_basis = "cartesian",
                ))
                
            except:
                if not noraise:
                    raise
                else:
                    return cells
        
        return cells
    
    @tag_method("unit-cell", take_file = True)
    def __unit_cells_silent__(self, f):
        # Search for an input file
        directory = os.path.dirname(f.name)
        file_names = list(os.path.join(directory,i) for i in os.listdir(directory))
        for name in file_names:
            if os.path.isfile(name):
                with open(name, "r") as f:
                    if Input in guess_parser(f):
                        try:
                            c = Input(f.read()).unitCell()
                            if c.size() == self.nat():
                                return self.unitCells(c, noraise = True)
                        except:
                            pass
        
        raise ParseError("Could not locate corresponding input file")
                        
    def populations(self):
        """
        Retrieves Mulliken populations during scf process.
        
        Returns:
        
            A numpy array where the first index corresponds to
            iteration number and the second one is atomic ID.
        """
        self.parser.reset()
        result = []
        
        while self.parser.present("NormRD"):
            
            self.parser.skip(re.compile(r"\*{19} MD=\s*\d*\s*SCF=\s*\d*\s*\*{19}"))
            self.parser.goto(re.compile(r"\n\s*"+re_int))
            self.parser.nextLine()
            
            c = []
            
            while self.parser.closest(("Sum of MulP",cre_int)) == 1:
                
                self.parser.skip("sum")
                c.append(self.parser.nextFloat())
                self.parser.nextLine()
            
            self.parser.skip("NormRD")
            result.append(c)
            
        return numpy.array(result)

    def solvers(self):
        """
        Retrieves the solver used for each iteration.
        
        Returns:
        
            A list of solver names.
        """
        self.parser.reset()
        result = []
        
        while self.parser.present("NormRD"):
            
            self.parser.skip(re.compile(r"\*{19} MD=\s*\d*\s*SCF=\s*\d*\s*\*{19}"))
            self.parser.goto("_DFT>")
            self.parser.startOfLine()
            result.append(self.parser.nextMatch(cre_word))
            self.parser.skip("NormRD")
            
        return result
        
    def convergence(self):
        """
        Retrieves convergence error values.
        
        Returns:
        
            A numpy array of convergence errors.
        """
        self.parser.reset()
        result = []
        
        while self.parser.present("NormRD"):
            
            self.parser.skip("NormRD")
            result.append(self.parser.nextFloat())
            
        return numpy.array(result)

class Bands(AbstractParser):
    """
    Class for parsing band structure from openmx.Band file.
    
    Args:
    
        data (string): contents of OpenMX Band file
    """
        
    @staticmethod
    def valid_filename(name):
        return name.endswith(".Band")
        
    def fermi(self):
        """
        Retrieves Fermi energy.
        
        Returns:
        
            Fermi energy.
        """
        self.parser.reset()
        p = self.parser
        
        p.nextInt()
        p.nextInt()
        return p.nextFloat()*2*numericalunits.Ry
        
    def captions(self):
        """
        Retrieves user-specified K-point captions.
        
        Returns:
        
            A dict with k-point number - k-point caption pairs.
        """
        self.parser.reset()
        p = self.parser
        
        p.nextLine(2)

        # nk = number of K points
        npath = p.nextInt()
        nk = 0
        result = {}
        
        for i in range(npath):
            
            n = p.nextInt()
            fr = p.nextFloat(3)
            to = p.nextFloat(3)
            fr_c = p.nextMatch(cre_nonspace)
            to_c = p.nextMatch(cre_nonspace)
            result[nk] = fr_c
            nk += n
            result[nk-1] = to_c
            p.nextLine()
            
        return result

    @band_structure
    def bands(self):
        """
        Retrieves bands.
        
        Returns:
        
            A UnitCell object with band energies.
        """
        fermi = self.fermi()
        
        self.parser.reset()
        p = self.parser
        
        bands = p.nextInt()
        p.nextInt()
        p.nextFloat()
        shape = 2*math.pi*p.nextFloat((3,3))/numericalunits.aBohr

        # nk = number of K points
        npath = p.nextInt()
        nk = 0
        for i in range(npath):
            nk += p.nextInt()
            p.nextLine()
        
        # initialize    
        coordinates = []
        values = []
        
        for i in range(nk):
            
            bands = p.nextInt()
            coordinates.append(p.nextFloat(3))
            p.nextLine()
            values.append(2*numericalunits.Ry*p.nextFloat(bands))
            p.nextLine()
            
        return UnitCell(
            Basis(shape, meta = {"Fermi":self.fermi(), "special-points":self.captions()}),
            coordinates,
            values,
        )

class Transmission(AbstractParser):
    """
    Class for parsing transmission from openmx.tran file.
    
    Args:
    
        data (string): contents of openmx.tran file
    """
        
    @staticmethod
    def valid_filename(name):
        l = name.rfind(".tran")
        
        if l == -1:
            return False
        print name[l+6:]
        return not (re.match(r"[\d+]_[\d+]\Z",name[l+5:]) is None)
        
    
    @staticmethod
    def valid_header(header):
        return "The unit of current is given by eEh/bar{h}" in header
        
    def __table__(self):
        """
        Reads the data as a table.
        
        Returns:
        
            All numbers presented in a numpy 2D array.
        """
        self.parser.reset()
        
        # Skip comments
        while self.parser.present("#"):
            self.parser.nextLine()
        
        result = []
        while self.parser.present(cre_float):
            result.append(self.parser.nextFloat("\n"))
            self.parser.nextLine()
            
        return numpy.array(result)
        
    def __spin__(self):
        """
        Retrieves integer corresponding to spin treatment.
        
        Returns:
        
            SpinP_switch as reported in the file.
        """
        self.parser.reset()
        return self.parser.intAfter("spinp_switch")

    def total(self):
        """
        Retrieves total transmission 1D array.
        
        Returns:
        
            A numpy array containing total transmission values with
            imaginary part discarded.
        """
        s = self.__spin__()
        table = self.__table__()
        if s == 0:
            return table[:,5]+table[:,7]
        elif s == 3:
            return table[:,5]
        else:
            raise ValueError("Unrecognized spinp_switch: {:d}".format(s))
        
    def energy(self):
        """
        Retrieves energy points of computed transmission.
        
        Returns:
        
            A numpy array containing energies with imaginary part discarded.
        """
        return self.__table__()[:,3]*numericalunits.eV

# Lower case versions
input = Input
output = Output
bands = Bands
transmission = Transmission
