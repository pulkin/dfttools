"""
Parsing `Quantum Espresso <http://www.quantum-espresso.org/>`_ files.
"""
import math
import re

import numpy
import numericalunits

from .generic import parse, cre_varName, cre_word, cre_float, cre_quotedText, re_float, cre_int, ParseError, AbstractParser
from ..simple import band_structure, unit_cell
from ..types import UnitCell, Basis

class Bands(AbstractParser):
    """
    Class for parsing output files created by bands.x binary of Quantum
    Espresso package.
    
    Args:
    
        data (str): string with the contents of the bands.x output file.
    """
    
    @staticmethod
    def valid_header(header):
        return "&plot" in header and "nbnd" in header and "nks" in header
        
    def nk(self):
        """
        Retrieves number of k points from the output file header.
        
        Returns:
        
            Integer number of k points.
        """
        
        self.parser.reset()
        self.parser.skip("nks=")
        return self.parser.nextInt()
        
    def ne(self):
        """
        Retrieves number of bands from the output file header.
        
        Returns:
        
            Integer number of bands.
        """
        
        self.parser.reset()
        self.parser.skip("nbnd=")
        return self.parser.nextInt()
    
    def bands(self, basis):
        """
        Retrieves the band structure data.
        
        Args:
        
            basis (types.Basis): the reciprocal unit cell of the band
            structure.
            
        Returns:
        
            A unit cell containing band structure data.
        """
        
        nk = self.nk()
        ne = self.ne()
        self.parser.reset()
        
        coordinates = numpy.zeros((nk,3))
        values = numpy.zeros((nk,ne))
        
        for i in range(nk):
            self.parser.nextLine()
            coordinates[i,:] = self.parser.nextFloat(3)
            values[i,:] = self.parser.nextFloat(ne)*numericalunits.eV
        
        return UnitCell(basis, coordinates, values)

class Output(AbstractParser):
    """
    Class for parsing output files created by pw.x binary of Quantum
    Espresso package.
    
    Args:
    
        data (str): string with the contents of the output file.
    """
        
    @staticmethod
    def valid_header(header):
        return "Program PWSCF" in header
        
    def success(self):
        """
        Checks for success signature in the end of the file.
        
        Returns:
        
            True if the signature is present.
        """
        return 'JOB DONE.' in self.data
        
    def routineError(self):
        """
        Checks "error in routine" entry in the file.
        
        Returns:
        
            String with textual information about the error. Returns
            None if no error recorded.
        """
        result = re.findall('Error in routine.*\n *(.*)\n',self.data)
        if len(result)==0:
            return None
        else:
            return result[0]
        
    def scf_accuracy(self):
        """
        Retrieves scf convergence history.
        
        Returns:
        
            A numpy array containing estimated errors after all scf
            steps during calculations. The energies are given in **eV**.
        """
        self.parser.reset()
        result = []
        
        while self.parser.present("estimated scf accuracy"):
            self.parser.skip("estimated scf accuracy")
            result.append(self.parser.nextFloat())
            
        return numpy.array(result)*numericalunits.Ry
        
    def scf_steps(self):
        """
        Retrieved number of scf steps.
        
        Returns:
        
            A numpy array containing numbers of consequetive scf steps
            performed to reach convergences.
        """
        return numpy.array(re.findall(r'\s*convergence has been achieved in\s+([-+]?\d+) iterations',self.data),dtype = numpy.int)
        
    def scf_failed(self):
        """
        Checks for "convergence NOT achieved" signature.
        
        Returns:
        
            True if the signature is present.
        """
        return 'convergence NOT achieved' in self.data
        
    def fermi(self):
        """
        Retrieves Fermi energies.
        
        Returns:
        
            A numpy array containing Fermi energies for each MD step.
        """
        result = []
        self.parser.reset()
        
        while True:
            
            parseMode = self.parser.closest(("End of self-consistent calculation","End of band structure calculation"))
            
            if parseMode is None:
                break
            elif parseMode == 0:
                self.parser.skip("End of self-consistent calculation")
            elif parseMode == 1:
                self.parser.skip("End of band structure calculation")
            
            x = self.parser.closest(("the Fermi energy is","highest occupied level","End of self-consistent calculation","End of band structure calculation"))
            
            if x == 0:
                self.parser.skip("the Fermi energy is")
                result.append(self.parser.nextFloat()*numericalunits.eV)
                
            elif x == 1:
                self.parser.skip("highest occupied level")
                result.append(self.parser.nextFloat()*numericalunits.eV)
            
            elif x == 2 or x == 3:
                result.append(None)
                
            else:
                break
            
        return numpy.array(result)

    def force(self):
        """
        Retrieves total force.
        
        Returns:
        
            A numpy array containing total forces for each
            self-consistent calculation.
        """
        self.parser.reset()
        result = []
        
        while self.parser.present("Total force ="):
            self.parser.skip("Total force =")
            result.append(self.parser.nextFloat())
            
        return numpy.array(result)*numericalunits.Ry/numericalunits.aBohr
        
    def total(self):
        """
        Retrieves total energies.
        
        Returns:
        
            A numpy array containing total energies for each
            self-consistent calculation.
        """
        self.parser.reset()
        result = []
        
        while self.parser.present("!    total energy"):
            self.parser.skip("!    total energy")
            result.append(self.parser.nextFloat())
            
        return numpy.array(result)*numericalunits.Ry
        
    def threads(self):
        """
        Retrieves the number of MPI threads.
        
        Returns:
        
            A number of MPI threads for this calculation as an integer.
        """
        return int(re.findall(r"running on\s+(\d+) processors",self.data)[-1])
        
    def time(self):
        """
        Retrieves cpu times.
        
        Returns:
        
            Time stamps measured by Quantum Espresso in a numpy array.
        """
        self.parser.reset()
        result = []
        
        while self.parser.present("total cpu time spent up to now is"):
            self.parser.skip("total cpu time spent up to now is")
            result.append(self.parser.nextFloat())
            
        return numpy.array(result)
        
    def alat(self):
        """
        Retrieves QE "alat" length.
        
        Returns:
        
            The first record of "alat" in **m** as float.
        """
        self.parser.reset()
        self.parser.skip("lattice parameter (alat)")
        alat = self.parser.nextFloat()
        
        # Determine alat in more precise way, if possible
        self.parser.skip("celldm(1)=")
        alat_precise = self.parser.nextFloat()
        if alat_precise != 0:
            return alat_precise*numericalunits.aBohr
        else:
            return alat*numericalunits.aBohr
    
    @unit_cell
    def unitCells(self):
        """
        Retrieves atomic position data.
        
        Returns:
        
            A set of all cells found with atomic coordinates in **m**.
            In ``Cell.values`` chemical captions are stored.
        """
        result = []
        
        alat = self.alat()
        
        self.parser.reset()
        
        # Parse initial unit cell
        self.parser.skip("number of atoms/cell")
        n = self.parser.nextInt()
        
        self.parser.skip("crystal axes: (cart. coord. in units of alat)")
        shape = self.parser.nextFloat((3,4))[:,1:]*alat
        basis = Basis(shape)
        
        self.parser.skip("Cartesian axes")
        coordinates = numpy.zeros((n,3))
        captions = numpy.zeros(n,dtype = 'S2')
        self.parser.nextLine(3)
        
        for i in range(n):
            
            self.parser.nextInt()
            captions[i] = self.parser.nextMatch(cre_word)
            self.parser.skip("=")
            coordinates[i,:] = self.parser.nextFloat(3)
            
        coordinates *= alat
        result.append(UnitCell(basis, coordinates, captions, c_basis = "cartesian"))
        
        # Parse MD steps
        while self.parser.present("ATOMIC_POSITIONS"):
            
            coordinates = numpy.zeros((n,3))
            captions = numpy.zeros(n,dtype = 'S2')
            
            # Check if vcr steps are present
            if self.parser.present("CELL_PARAMETERS") and (self.parser.distance("CELL_PARAMETERS")<self.parser.distance("ATOMIC_POSITIONS")):
                
                self.parser.skip("CELL_PARAMETERS")
                alat = self.parser.nextFloat()*numericalunits.aBohr
                shape = self.parser.nextFloat((3,3))*alat
                basis = Basis(shape)
            
            # Parse atomic data
            self.parser.skip("ATOMIC_POSITIONS")
            units = self.parser.nextMatch(cre_word)
            
            for i in range(n):
                
                captions[i] = self.parser.nextMatch(cre_word)
                coordinates[i,:] = self.parser.nextFloat(3)
                self.parser.nextLine()
                
            if units == "crystal":
                result.append(UnitCell(basis, coordinates, captions))
            elif units == "alat":
                result.append(UnitCell(basis, coordinates*alat, captions, c_basis = "cartesian"))
            elif units == "bohr":
                result.append(UnitCell(basis, coordinates*numericalunits.aBohr, captions, c_basis = "cartesian"))
            else:
                raise ParseError("Unknown units: %s" % units)
                
        return result
    
    def bands(self, skipVCRelaxException = False):
        """
        Retrieves bands.
        
        Kwargs:
        
            skipVCRelaxException (bool): forces to skip variable cell
            relaxation exception. In this very special case no
            reciprocal lattice vectors are provided for the new cells
            in the output file.
                
        Returns:
        
            A set of Cell objects with bands data stored in ``Cell.values``.
            Specifically, ``Cell.values`` is a n by m array where n is a
            number of k points and m is a number of bands.
            
        Raises:
        
            Exception: if a variable cell calculation data found.
        """
        fermi = self.fermi()
        self.parser.reset()
        if self.parser.present("new lattice vectors") and not skipVCRelaxException:
            raise Exception("Variable cell relaxation output detected. "+
                "No reciprocal lattice vectors found for relaxed cells. "+
                "To skip this exception and write kpoints in old basis "+
                "set skipVCRelaxExeption to True.")
        
        alat = self.alat()
        
        self.parser.reset()
        self.parser.skip("reciprocal axes: (cart. coord. in units 2 pi/alat)")
        shape = self.parser.nextFloat((3,4))[:,1:]*2*math.pi/alat
        
        self.parser.skip("number of k points=")
        n_kp = self.parser.nextInt()
        
        if self.parser.present("cryst. coord."):
            
            parseMode_kp = 0
            self.parser.skip("cryst. coord.")
            kpoints = self.parser.nextFloat((n_kp,5))[:,1:4]
            
        elif self.parser.present("cart. coord. in units 2pi/alat"):
            
            parseMode_kp = 1
            self.parser.skip("cart. coord. in units 2pi/alat")
            kpoints = self.parser.nextFloat((n_kp,5))[:,1:4]
            
        else:
            raise Exception("No kpoint data found in the file.")
        
        bandStructures = []
        
        while True:
            
            parseMode = self.parser.closest(("End of self-consistent calculation","End of band structure calculation"))
            
            if parseMode is None:
                break
            elif parseMode == 0:
                self.parser.skip("End of self-consistent calculation")
            elif parseMode == 1:
                self.parser.skip("End of band structure calculation")
                
            energies = []
            
            for i in range(n_kp):
                
                self.parser.skip("k =")
                self.parser.nextLine(2)
                sub_energies = []
                
                while self.parser.closest((cre_float,"\n\n")) == 0:
                    sub_energies.append(self.parser.nextFloat()*numericalunits.eV)
                    
                energies.append(sub_energies)
            
            if parseMode_kp == 0:
                bandStructures.append(UnitCell(Basis(shape), kpoints, energies))
            else:
                bandStructures.append(UnitCell(Basis(shape), kpoints*2*math.pi/alat, energies, c_basis = "cartesian"))
                
            if len(fermi)>0:
                bandStructures[-1].meta["Fermi"] = fermi[0]
                fermi = fermi[1:]
            
        return bandStructures
    
    @band_structure
    def __bands_silent__(self):
        return self.bands(skipVCRelaxException = True)
        
class Proj(AbstractParser):
    """
    Class for parsing output files created by projwfc.x binary of
    Quantum Espresso package.
    
    Args:
    
        data (str): string with the contents of the output file.
    """
        
    @staticmethod
    def valid_header(header):
        return "Program PROJWFC" in header
        
    def basis(self):
        """
        Retrieves the localized basis set.
        
        Returns:
        
            A numpy array of records:
            
            * state (int): ID of the state as provided by Quantum Espresso;
            * atom (int): ID of particular atom in the unit cell;
            * atomName (str): chemical caption;
            * wfc (int): particular wave function ID;
            * m,l (float): quantum numbers for collinear calculation;
            * j,m_j,l (float): quantum numbers for non-collinear calculation.
        """
        self.parser.reset()
        
        if self.parser.present("Calling projwave ...."):
            
            self.parser.skip("Calling projwave ....")
            
            dataType = numpy.dtype([
                ('state',numpy.int64,1),
                ('atom',numpy.int64,1),
                ('atomName', numpy.str_, 2),
                ('wfc', numpy.int64, 1),
                ('l', numpy.float64, 1),
                ('m', numpy.float64, 1)
            ])
            
            result = []
            
            while self.parser.present("state #"):
                
                self.parser.skip("state #")
                result.append((
                    self.parser.nextInt(),
                    self.parser.nextInt(),
                    self.parser.nextMatch(cre_word),
                    self.parser.nextInt(),
                    self.parser.nextFloat(),
                    self.parser.nextFloat()
                ))
                
            states = numpy.array(result, dtype = dataType)
                
        elif self.parser.present("Calling projwave_nc ...."):
            
            self.parser.skip("Calling projwave_nc ....")
            
            dataType = numpy.dtype([
                ('state',numpy.int64,1),
                ('atom',numpy.int64,1),
                ('atomName', numpy.str_, 2),
                ('wfc', numpy.int64, 1),
                ('j', numpy.float64, 1),
                ('l', numpy.float64, 1),
                ('m_j', numpy.float64, 1)
            ])
            
            result = []
            
            while self.parser.present("state #"):
                
                self.parser.skip("state #")
                result.append((
                    self.parser.nextInt(),
                    self.parser.nextInt(),
                    self.parser.nextMatch(cre_word),
                    self.parser.nextInt(),
                    self.parser.nextFloat(),
                    self.parser.nextFloat(),
                    self.parser.nextFloat()
                ))
                
            states = numpy.array(result, dtype = dataType)
        
        else:
            raise Exception("Unknown projwfc output file.")
            
        return states
        
    def weights(self, lower = 0, upper = None):
        """
        Retrieves projection weights onto localized basis set.
        
        Kwargs:
        
            bands (tuple
        
        Returns:
        
            A k by n by m numpy array with weights.
            
            * k is a number of k points
            * n is a number of bands
            * m is a localized basis set size
        """
        
        basisSize = self.basis().shape[0]
        
        self.parser.reset()
        self.parser.skip("Calling projwave")
        
        projections = []
        
        while self.parser.present("k ="):
            
            self.parser.skip("k =")
            projections_k = []
            
            for i in range(lower):
                self.parser.skip("==== e(")
                    
            while self.parser.closest(("==== e(","k =")) == 0:
                
                self.parser.skip("==== e(")
                self.parser.nextLine()
                rawData = self.parser.nextFloat("|psi|^2")
                
                projections_ke = numpy.zeros(basisSize)
                for i in range(rawData.shape[0]/2):
                    projections_ke[rawData[2*i+1]-1] = rawData[2*i]
                projections_k.append(projections_ke)
                
                if not upper is None and len(projections_k) >= (upper-lower):
                    break
                
            projections.append(projections_k)
            
        return numpy.array(projections)

class Cond(AbstractParser):
    """
    Class for parsing output files created by pwcond.x binary of Quantum
    Espresso package.
    
    Args:
    
        data (str): string with the contents of the output file.
    """
        
    @staticmethod
    def valid_header(header):
        return "Program PWCOND" in header
    
    def transmission(self, kind = "resolved"):
        """
        Retrives transmission data from pwcond output file.
        
        Kwargs:
        
            kind (str): either "resolved", "total", "states_in" or
                "states_out".
            
                * resolved: retrieves transmisson matrix elements btw
                  pairs of states;
                * total: retrieves total transmission for all incoming
                  states;
                * states_in, states_out: retrieves only incoming or
                  outgoing states without forming pairs and obtaining
                  transmissions.

        .. warning::
                  
            The "resolved" mode essentially picks all transmission
            matrix elements available in the input. Therefore it
            will not record incident states without corresponding
            outgoing states. However these states will show up in
            "total" regime with zero transmission.
                  
        Returns:
        
            A numpy array of records with states and transmission:
            
            * energy (float): energy of the state in **eV**;
            * kx,ky (float): x and y components of k-vectors;
            * incoming (float): z component of k-vector of incoming
              state (only for kind == "resolved" or kind == "total" or
              kind == "states_in");
            * outgoing (float): z component of k-vector of outgoing
              state (only for kind == "resolved" or kind == "states_out");
            * transmission (float): corresponding transmission matrix
              element or total transmission (only for kind == "resolved"
              or kind == "total").
                
            The k vector projections are given in units of reciprocal
            lattice.
        """
        
        if kind == "resolved":
            dataType = numpy.dtype([
                ('energy',numpy.float64,1),
                ('kx',numpy.float64,1),
                ('ky',numpy.float64,1),
                ('incoming',numpy.complex64,1),
                ('outgoing', numpy.complex64,1),
                ('transmission', numpy.float64, 1),
            ])
        elif kind == "total":
            dataType = numpy.dtype([
                ('energy',numpy.float64,1),
                ('kx',numpy.float64,1),
                ('ky',numpy.float64,1),
                ('incoming',numpy.complex64,1),
                ('transmission', numpy.float64, 1),
            ])
        elif kind == "states_in":
            dataType = numpy.dtype([
                ('energy',numpy.float64,1),
                ('kx',numpy.float64,1),
                ('ky',numpy.float64,1),
                ('incoming',numpy.complex64,1),
            ])
        elif kind == "states_out":
            dataType = numpy.dtype([
                ('energy',numpy.float64,1),
                ('kx',numpy.float64,1),
                ('ky',numpy.float64,1),
                ('outgoing',numpy.complex64,1),
            ])
        else:
            raise ParseError("Unknown kind: '%s'; should be either 'total', 'resolved', 'states_in' or 'states_out'." % kind)
        
        result = []
        self.parser.reset()
        previous = numpy.zeros(1,dtype = dataType)
        
        try:
            
            while self.parser.present("---  E-Ef"):
                
                self.parser.skip("---  E-Ef")
                
                previous['energy'] = self.parser.nextFloat()
                previous['kx'] = self.parser.nextFloat()
                previous['ky'] = self.parser.nextFloat()
                
                channels = {"left":{"left":None, "right":None},
                    "right":{"left":None, "right":None}}
                
                for lead in channels.keys():
                    
                    if self.parser.closest(("Nchannels of the %s tip =" % lead,"---  E-Ef")) == 0:
                        
                        self.parser.save()
                        self.parser.skip("Nchannels of the %s tip =" % lead)
                        numberOfChannels = self.parser.nextInt()
                    
                        for direction in channels[lead].keys():
                        
                            self.parser.save()
                            self.parser.skip("%s moving states:" % direction)
                            self.parser.nextLine(2)
                            states = self.parser.nextFloat((numberOfChannels,3))
                            channels[lead][direction] = states[:,0] + 1.j*states[:,1]
                            self.parser.pop()
                        
                        self.parser.pop()
                
                # If leads are the same
                if channels["left"]["left"] is None:
                    channels["left"]["left"] = channels["right"]["left"]
                if channels["left"]["right"] is None:
                    channels["left"]["right"] = channels["right"]["right"]
                if channels["right"]["left"] is None:
                    channels["right"]["left"] = channels["left"]["left"]
                if channels["right"]["right"] is None:
                    channels["right"]["right"] = channels["left"]["right"]
                
                if kind == "states_in":
                    # Just list incoming states
                    
                    for kz in channels["left"]["right"]:
                        
                        previous["incoming"] = kz
                        result.append(previous)
                        previous = previous.copy()
                    
                elif kind == "states_out":
                    # Just list outgoing states
                    
                    for kz in channels["right"]["right"]:
                        
                        previous["outgoing"] = kz
                        result.append(previous)
                        previous = previous.copy()
                    
                elif kind == "total" or kind == "resolved":
                    # Read transmissions
                    
                    self.parser.skip("to transmit")
                    
                    entry = self.parser.closest(("-->","Total T_j, R_j =","E-Ef(ev), T ="))
                    while (entry == 0) or (entry == 1):
                        
                        if entry == 0:
                            
                            self.parser.goto("-->")
                            self.parser.startOfLine()
                            
                            initial = self.parser.nextInt()-1
                            previous["incoming"] = channels["left"]["right"][initial]
                            
                            final = self.parser.nextInt()-1
                            
                            if final<len(channels["right"]["right"]) and kind == "resolved":
                                   
                                t = self.parser.nextFloat()
                                
                                previous["outgoing"] = channels["right"]["right"][final]
                                previous["transmission"] = t
                                result.append(previous)
                                previous = previous.copy()
    
                            self.parser.nextLine()
                            
                        elif entry == 1:
                            
                            self.parser.skip("Total T_j, R_j =")
                            
                            t = self.parser.nextFloat()
                            self.parser.nextLine()
                            
                            if kind == "total":
                                
                                previous["transmission"] = t
                                result.append(previous)
                                previous = previous.copy()
                                
                        entry = self.parser.closest(("-->","Total T_j, R_j =","E-Ef(ev), T ="))
                    
                    # If no outgoing states found and total transmission is
                    # required adds incoming states with zero transmission
                    
                    if kind == "total" and len(channels["right"]["right"]) == 0:
                        
                        for incoming in channels["left"]["right"]:
                            
                            previous["incoming"] = incoming
                            previous["transmission"] = 0
                            result.append(previous)
                            previous = previous.copy()
                        
        except:
            
            if len(result) == 0:
                
                raise
                
        return numpy.concatenate(result)

class Input(AbstractParser):
    """
    Class for parsing input file for pw.x binary of a Quantum Espresso
    package.
    
    Args:
    
        data (str): string with the contents of the input file.
    """

    def __init__(self, data):
        lines = []
        for line in data.split('\n'):
            if not line.strip().startswith('!'):
                lines.append(line)
        self.data = "\n".join(lines)
        self.parser = parse(self.data)
        
    @staticmethod
    def valid_header(header):
        l = header.lower()
        return "&control" in l or "&system" in l or "&electrons" in l or "&ions" in l
        
    def namelists(self):
        """
        Retrieves all namelists.
        
        Returns:
        
            A dictionary representing this namelist.
        """
        self.parser.reset()
        result = {}
        
        while self.parser.present("&"):
            
            self.parser.skip("&")
            
            nl = self.parser.nextMatch(cre_varName).lower()
            result[nl] = {}
            
            while True:
                
                nxt = self.parser.closest(("/", cre_varName))
                
                if nxt == 0 or nxt == -1:
                    break
                    
                name = self.parser.nextMatch(cre_varName).lower()
                self.parser.skip("=")
                dataType = self.parser.closest(("false","true",cre_float,cre_quotedText))
                if dataType == 0:
                    result[nl][name] = False
                    self.parser.skip("false")
                elif dataType == 1:
                    result[nl][name] = True
                    self.parser.skip("true")
                elif dataType == 2:
                    result[nl][name] = self.parser.nextFloat()
                elif dataType == 3:
                    result[nl][name] = self.parser.nextMatch(cre_quotedText)[1:-1]
                else:
                    raise Exception("Could not retrieve value for {}.{}".format(nl,name))
                    
                    
            self.parser.skip("/")
                    
        return result
        
    @unit_cell
    def unitCell(self):
        """
        Retrieves a unit cell from this input file.
        
        Returns:
        
            A unit cell with atomic coordinates.
        """
        nl = self.namelists()
        
        units_dict = {
            "angstrom": numericalunits.angstrom,
            "bohr": numericalunits.aBohr,
            "alat": nl["system"]["celldm(1)"] if "celldm(1)" in nl["system"] else None,
        }
        
        ibrav = nl["system"]["ibrav"]
        if ibrav == 14:
            
            shape = list(nl["system"]["celldm(%i)"% (i+1)] for i in range(6))
            shape[0] *= numericalunits.aBohr
            shape[1] *= shape[0]
            shape[2] *= shape[0]
            basis = Basis(shape, kind = 'triclinic')
            
        elif ibrav == 0:

            self.parser.reset()
            self.parser.skip("cell_parameters")
            units = self.parser.nextMatch(cre_word)
            vectors = self.parser.nextFloat(n = (3,3))*units_dict[units]
                
            basis = Basis(vectors)
            
        else:
            raise NotImplementedError("Cell recovery not implemented for ibrav = {:d}".format(int(ibrav)))
        
        self.parser.reset()
        self.parser.skip("atomic_positions")
        units = self.parser.nextMatch(cre_word)
        coordinates = numpy.zeros((nl["system"]["nat"],3))
        statics = numpy.ones(coordinates.shape)
        values = numpy.zeros(coordinates.shape[0],"S2")
        
        for i in range(coordinates.shape[0]):
            
            values[i] = self.parser.nextMatch(cre_word)
            coordinates[i,:] = self.parser.nextFloat(3)
            
            if self.parser.closest(("\n",cre_int)) == 1:
                
                statics[i,:] = self.parser.nextFloat(3)
                
            self.parser.nextLine()
            
        if units == "alat":
            return UnitCell(basis, coordinates*alat, values, c_basis = "cartesian")
        elif units == "bohr":
            return UnitCell(basis, coordinates*numericalunits.aBohr, values, c_basis = "cartesian")
        elif units == "angstrom":
            return UnitCell(basis, coordinates*numericalunits.angstrom, values, c_basis = "cartesian")
        elif units == "crystal":
            return UnitCell(basis, coordinates, values)

# Lower case versions
bands = Bands
output = Output
cond = Cond
proj = Proj
input = Input
