"""
Parsing `OpenMX <http://openmx-square.org/>`_ files.
"""
import json
import os
import re
from itertools import chain

import numericalunits
import numpy
import os.path

from .generic import cre_var_name, cre_word, cre_non_space, re_int, cre_int, cre_float, AbstractTextParser, \
    AbstractJSONParser, IdentifiableParser, ParseError
from .native_openmx import openmx_bands_bands
from ..simple import band_structure, unit_cell, guess_parser, tag_method
from ..utypes import CrystalCell, BandsPath


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

    result["energies"] *= numericalunits.Hartree

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

        if i > 0:
            if not numpy.all(p["bands"] == p2["bands"]):
                raise ValueError("Different bands reported at k #0 and k #{:d}".format(i))
            for k in chain(p["basis"].keys(), p2["basis"].keys()):
                if k not in p["basis"] or k not in p2["basis"]:
                    raise ValueError("The basis description '{}' is missing for k #0 or k #{:d}".format(k, i))
                if not numpy.all(p["basis"][k] == p2["basis"][k]):
                    raise ValueError("The basis description '{}' is different for k #0 and k #{:d}".format(k, i))

        for c in collected.keys():
            collected[c].append(p[c])

    result = parsed[0]
    del result["k-id"]
    for c in collected.keys():
        result[c] = numpy.array(collected[c])

    return result


class JSON_DOS(AbstractJSONParser, IdentifiableParser):
    """
    Parses JSON with OpenMX density of states.
    
    Args:
    
        data (str): contents of OpenMX JSON DoS file.
    """

    @staticmethod
    def valid_header(header):
        return "openmx-dos-negf" in header

    @staticmethod
    def valid_filename(name):
        return name.endswith(".Dos.json")

    def __init__(self, data):
        super(JSON_DOS, self).__init__(data)
        self.__set_units__("energy", numericalunits.Hartree)
        self.__set_units__("DOS", 1. / numericalunits.Hartree)

        for field in self.json["basis"].keys():
            self.json["basis"][field] = numpy.array(self.json["basis"][field])

    def basis(self):
        """
        Retrieves the basis set for density weights.
        
        Returns:
        
            A dict contatining basis description.
        """
        return self.json["basis"]

    def weights(self):
        """
        Retrieves the densities.
        
        Returns:
        
            Densities in a 4D array with the following index order:
            * ky
            * kz
            * energy
            * state
        """
        return self.json["DOS"]

    def energies(self):
        """
        Retrieves corresponding energies.
        
        Returns:
        
            A 1D array with energy values.
        """
        return numpy.linspace(self.json["energy"][0], self.json["energy"][1], self.json["DOS"].shape[2])

    def __k__(self, index):
        n = self.json["DOS"].shape[index]
        k = numpy.linspace(0, 1, n, endpoint=False)
        k -= k[-1] / 2
        return k

    def ky(self):
        """
        Retrieves values of ky.
        
        Returns:
        
            Values of ky.
        """
        return self.__k__(0)

    def kz(self):
        """
        Retrieves values of kz.
        
        Returns:
        
            Values of kz.
        """
        return self.__k__(1)


class Input(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing parameter values from OpenMX input files.
    
    Args:
    
        data (str): contents of OpenMX input file
    """

    @staticmethod
    def valid_header(header):
        l = header.lower()
        return "definition.of.atomic.species" in l

    @staticmethod
    def valid_filename(name):
        raise NotImplementedError

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
        return self.parser.match_after(parameter, cre_word)

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
        return self.parser.match_after(parameter, cre_non_space)

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
        return self.parser.float_after(parameter)

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
        return self.parser.int_after(parameter)

    @unit_cell
    def unitCell(self, l=None, r=None, tolerance=1e-12):
        """
        Retrieves atomic position data.
        
        Kwargs:
        
            l,r (UnitCell): left lead and right lead cells.
            This information is required for parsing the cell from NEGF
            calculation input file;
            
            tolerance (float): a tolerance for comparing atomic position
            data from the keywords and from the file itself in ``aBohr``.
        
        Returns:
        
            An input unit cell.
            
        Raises:
        
            ValueError: left and right lead cells are not specified for
            NEGF input file.
        """

        self.parser.reset()

        if not self.parser.present("<Atoms.UnitVectors"):
            if l is None or r is None:
                raise ValueError(
                    "The input file does not specify unit cell dimensions (NEGF calculation) but the left and right leads are not specified")
            no_vectors = True

        else:
            no_vectors = False

        n = self.getInt("Atoms.Number")
        units = self.getWord("Atoms.SpeciesAndCoordinates.Unit")

        self.parser.save()
        self.parser.skip("<Atoms.SpeciesAndCoordinates")
        coordinates = numpy.zeros((n, 3))
        values = []
        for i in range(n):
            self.parser.next_int()
            values.append(self.parser.next_match(cre_word))
            coordinates[i, :] = self.parser.next_float(3)
            self.parser.next_line()
        self.parser.pop()

        if units.lower() == "ang":
            coordinates *= numericalunits.angstrom
        elif units.lower() == "au":
            coordinates *= numericalunits.aBohr

        if no_vectors:

            nl = self.parser.int_after("LeftLeadAtoms.Number")
            nr = self.parser.int_after("RightLeadAtoms.Number")

            self.parser.skip("<LeftLeadAtoms.SpeciesAndCoordinates")
            coordinatesl = numpy.zeros((nl, 3))
            valuesl = []

            for i in range(nl):
                self.parser.next_int()
                valuesl.append(self.parser.next_match(cre_word))
                coordinatesl[i, :] = self.parser.next_float(3)
                self.parser.next_line()

            self.parser.reset()

            self.parser.skip("<RightLeadAtoms.SpeciesAndCoordinates")
            coordinatesr = numpy.zeros((nr, 3))
            valuesr = []

            for i in range(nr):
                self.parser.next_int()
                valuesr.append(self.parser.next_match(cre_word))
                coordinatesr[i, :] = self.parser.next_float(3)
                self.parser.next_line()

            self.parser.reset()

            if units.lower() == "ang":
                coordinatesl *= numericalunits.angstrom
                coordinatesr *= numericalunits.angstrom
            elif units.lower() == "au":
                coordinatesl *= numericalunits.aBohr
                coordinatesr *= numericalunits.aBohr

            dl = (l.cartesian() - coordinatesl).sum(axis=0) / l.size()
            delta_l = abs(l.cartesian() - coordinatesl - dl).max()

            dr = (r.cartesian() - coordinatesr).sum(axis=0) / r.size()
            delta_r = abs(r.cartesian() - coordinatesr - dr).max()

            if delta_l / numericalunits.aBohr > tolerance:
                raise ValueError(
                    "The atomic coordinates given by 'l' keyword and those present in the input file are different. Set the 'tolerance' keyword to at least {:e} to avoid this error.".format(
                        delta_l / numericalunits.aBohr))
            if delta_r / numericalunits.aBohr > tolerance:
                raise ValueError(
                    "The atomic coordinates given by 'r' keyword and those present in the input file are different. Set the 'tolerance' keyword to at least {:e} to avoid this error.".format(
                        delta_r / numericalunits.aBohr))

            shape = l.vectors.copy()
            shape[0, :] = dl - dr - l.vectors[0, :]
            coordinates = coordinates + dl - l.vectors[0, :]

        else:

            units_cell = self.parser.match_after("Atoms.UnitVectors.Unit", cre_word).lower()
            shape = self.parser.float_after("<Atoms.UnitVectors", n=(3, 3))

            if units_cell == "ang":
                shape *= numericalunits.angstrom
            elif units_cell == "au":
                shape *= numericalunits.aBohr

        return CrystalCell(
            shape,
            coordinates,
            values,
            c_basis=None if units.lower() == "frac" else "cartesian"
        )


class Output(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing parameter values from OpenMX output files.
    
    Args:
    
        data (string): contents of OpenMX output file
    """

    @staticmethod
    def valid_header(header):
        return "Welcome to OpenMX" in header and "T. Ozaki" in header

    @staticmethod
    def valid_filename(name):
        raise NotImplementedError

    def version(self):
        """
        Retrieves OpenMX version as reported in the output.
        
        Returns:
        
            OpenMX program version as string.
        """
        self.parser.reset()
        self.parser.skip(re.compile("Welcome to OpenMX\s+Ver\."))
        return self.parser.next_match(cre_var_name, n="\n")[0]

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
            result.append(self.parser.next_float() * numericalunits.Hartree)

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
        while self.parser.match_closest(("XYZ(ang)", "***")) == 0:
            self.parser.skip("XYZ(ang)")
            n += 1

        return n

    def unitCells(self, startingCell, noraise=False):
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
                shape = self.parser.next_float((3, 3)) * numericalunits.aBohr

                self.parser.skip("MD or geometry opt. at MD")
                self.parser.skip("maximum force")
                coordinates = []

                while self.parser.match_closest(("XYZ(ang)", "***")) == 0:
                    self.parser.skip("XYZ(ang)")
                    coordinates.append(self.parser.next_float(3) * numericalunits.angstrom)

                cells.append(CrystalCell(
                    shape,
                    coordinates,
                    startingCell.values,
                    c_basis="cartesian",
                ))

            except:
                if not noraise:
                    raise
                else:
                    return cells

        return cells

    @tag_method("unit-cell", take_file=True)
    def __unit_cells_silent__(self, f):
        # Search for an input file
        directory = os.path.dirname(f.name)
        file_names = list(os.path.join(directory, i) for i in os.listdir(directory))
        for name in file_names:
            if os.path.isfile(name):
                with open(name, "r") as f:
                    if Input in guess_parser(f):
                        try:
                            c = Input(f.read()).unitCell()
                            if c.size() == self.nat():
                                return self.unitCells(c, noraise=True)
                        except:
                            pass

        raise ParseError("Could not locate corresponding input file")

    def populations(self):
        """
        Retrieves Mulliken populations during scf process.
        
        Returns:
        
            A numpy array where the first index corresponds to
            iteration number and the second one is atomic ID. The
            populations are renormalized to reproduce the total charge.
        """
        self.parser.reset()
        result = []

        while self.parser.present("NormRD"):

            self.parser.skip(re.compile(r"\*{19} MD=\s*\d*\s*SCF=\s*\d*\s*\*{19}"))
            self.parser.goto(re.compile(r"\n\s*" + re_int))
            self.parser.next_line()

            c = []

            while self.parser.match_closest(("Sum of MulP", cre_int)) == 1:
                self.parser.skip("sum")
                c.append(self.parser.next_float())
                self.parser.next_line()

            self.parser.skip("total=")
            total = self.parser.next_float()

            self.parser.skip("NormRD")
            c = numpy.array(c)
            result.append(c * total / sum(c))

        return numpy.array(result)

    def neutral_charge(self):
        """
        Retrieves the number of valence electrons in the calculation for
        the charge neutral system.
        
        Returns:
        
            The number of electrons.
        """
        self.parser.reset()
        return self.parser.float_after("ideal(neutral)=")

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
            self.parser.rtn()
            result.append(self.parser.next_match(cre_word))
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
            result.append(self.parser.next_float())

        return numpy.array(result)


class Bands(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing band structure from openmx.Band file.
    
    Args:
    
        data (string): contents of OpenMX Band file
    """

    @staticmethod
    def valid_header(header):
        raise NotImplementedError

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

        p.next_int()
        p.next_int()
        return p.next_float() * numericalunits.Hartree

    def captions(self):
        """
        Retrieves user-specified K-point captions.
        
        Returns:
        
            A dict with k-point number - k-point caption pairs.
        """
        self.parser.reset()
        p = self.parser

        p.next_line(2)

        # nk = number of K points
        npath = p.next_int()
        nk = 0
        result = {}

        for i in range(npath):
            n = p.next_int()
            fr = p.next_float(3)
            to = p.next_float(3)
            fr_c = p.next_match(cre_non_space)
            to_c = p.next_match(cre_non_space)
            result[nk] = fr_c
            nk += n
            result[nk - 1] = to_c
            p.next_line()

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

        p.next_line()
        shape = p.next_float((3, 3)) / numericalunits.aBohr

        data = openmx_bands_bands(self.data)

        return BandsPath(
            shape,
            data[:, :3],
            data[:, 3:] * numericalunits.Hartree,
            fermi=fermi,
            meta={"special-points": self.captions()},
        )


class Transmission(AbstractTextParser, IdentifiableParser):
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
        return not (re.match(r"[\d+]_[\d+]\Z", name[l + 5:]) is None)

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
            self.parser.next_line()

        result = []
        while self.parser.present(cre_float):
            result.append(self.parser.next_float("\n"))
            self.parser.next_line()

        return numpy.array(result)

    def __spin__(self):
        """
        Retrieves integer corresponding to spin treatment.
        
        Returns:
        
            SpinP_switch as reported in the file.
        """
        self.parser.reset()
        return self.parser.int_after("spinp_switch")

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
            return table[:, 5] + table[:, 7]
        elif s == 3:
            return table[:, 5]
        else:
            raise ValueError("Unrecognized spinp_switch: {:d}".format(s))

    def energy(self):
        """
        Retrieves energy points of computed transmission.
        
        Returns:
        
            A numpy array containing energies with imaginary part discarded.
        """
        return self.__table__()[:, 3] * numericalunits.eV


# Lower case versions
input = Input
output = Output
bands = Bands
transmission = Transmission
