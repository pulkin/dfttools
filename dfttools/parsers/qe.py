"""
Parsing `Quantum Espresso <http://www.quantum-espresso.org/>`_ files.
"""
import math
import re

import numericalunits
import numpy

from .generic import parse, cre_var_name, cre_word, cre_float, cre_quotedText, cre_int, ParseError, \
    AbstractTextParser, IdentifiableParser
from .native_qe import qe_proj_weights, qe_scf_cell
from ..simple import band_structure, unit_cell, tag_method
from ..types import CrystalCell, BandsPath, RealSpaceBasis, element_type
from ..util import eV, eV_angstrom, K


class Bands(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing output files created by bands.x binary of Quantum
    Espresso package.

    Args:

        data (str): string with the contents of the bands.x output file.
    """

    @staticmethod
    def valid_header(header):
        return "&plot" in header and "nbnd" in header and "nks" in header

    @staticmethod
    def valid_filename(name):
        raise NotImplementedError

    def nk(self):
        """
        Retrieves number of k points from the output file header.

        Returns:

            Integer number of k points.
        """

        self.parser.reset()
        self.parser.skip("nks=")
        return self.parser.next_int()

    def ne(self):
        """
        Retrieves number of bands from the output file header.

        Returns:

            Integer number of bands.
        """

        self.parser.reset()
        self.parser.skip("nbnd=")
        return self.parser.next_int()

    @tag_method("basis-dependent")
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

        coordinates = numpy.zeros((nk, 3))
        values = numpy.zeros((nk, ne))

        for i in range(nk):
            self.parser.next_line()
            coordinates[i, :] = self.parser.next_float(3)
            values[i, :] = self.parser.next_float(ne)

        return BandsPath(basis, coordinates, values * numericalunits.eV)


class Output(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing output files created by pw.x binary of Quantum
    Espresso package.

    Args:

        data (str): string with the contents of the output file.
    """

    @staticmethod
    def valid_header(header):
        return "Program PWSCF" in header

    @staticmethod
    def valid_filename(name):
        raise NotImplementedError

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
        result = re.findall('Error in routine.*\n *(.*)\n', self.data)
        if len(result) == 0:
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
            result.append(self.parser.next_float())

        return numpy.array(result) * numericalunits.Ry

    def scf_steps(self):
        """
        Retrieved number of scf steps.

        Returns:

            A numpy array containing numbers of consequetive scf steps
            performed to reach convergences.
        """
        return numpy.array(re.findall(r'\s*convergence has been achieved in\s+([-+]?\d+) iterations', self.data),
                           dtype=numpy.int)

    def scf_failed(self):
        """
        Checks for "convergence NOT achieved" signature.

        Returns:

            True if the signature is present.
        """
        return 'convergence NOT achieved' in self.data

    def fermi(self, eps=numpy.finfo(float).eps):
        """
        Retrieves Fermi energies.
        Args:
            eps (float): machine epsilon for shifting the Fermi level from
            the highest occupied band energy;

        Returns:

            A numpy array containing Fermi energies for each MD step.
        """
        result = []
        self.parser.reset()

        while True:

            parseMode = self.parser.match_closest(("End of self-consistent calculation", "End of band structure calculation"))

            if parseMode is None:
                break
            elif parseMode == 0:
                self.parser.skip("End of self-consistent calculation")
            elif parseMode == 1:
                self.parser.skip("End of band structure calculation")

            x = self.parser.match_closest(("the Fermi energy is", "highest occupied level",
                                     "End of self-consistent calculation", "End of band structure calculation"))

            if x == 0:
                self.parser.skip("the Fermi energy is")
                result.append(self.parser.next_float() * numericalunits.eV)

            elif x == 1:
                self.parser.skip("highest occupied level")
                result.append(self.parser.next_float() * (1 + eps) * numericalunits.eV)

            elif x == 2 or x == 3:
                result.append(None)

            else:
                break

        return eV(result)

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
            result.append(self.parser.next_float() * numericalunits.Ry / numericalunits.aBohr)

        return eV_angstrom(result)

    def __next_forces__(self):
        self.parser.skip("Forces acting on atoms")
        self.parser.skip("atom")
        return eV_angstrom(self.parser.next_float("Total force").reshape(-1, 5)[:, 2:] * numericalunits.Ry / numericalunits.aBohr)

    def forces(self):
        """
        Retrieves forces per atom per iteration.
        Returns:
            A 3D numpy array with all force vectors.
        """
        self.parser.reset()
        result = []
        try:
            while True:
                result.append(self.__next_forces__())
        except StopIteration:
            pass
        return eV_angstrom(result)

    def __next_total__(self):
        c = self.parser.match_closest(("!    total energy", "final energy", "energy   new"))
        if c == 0:
            self.parser.skip("!    total energy")
            return eV(self.parser.next_float() * numericalunits.Ry)
        elif c == 1:
            self.parser.skip("final energy")
            return eV(self.parser.next_float() * numericalunits.Ry)
        else:
            raise StopIteration("No total energies are available")

    def total(self):
        """
        Retrieves total energies.

        Returns:

            A numpy array containing total energies for each
            self-consistent calculation.
        """
        self.parser.reset()
        result = []

        try:
            while True:
                result.append(self.__next_total__())
        except StopIteration:
            pass
        return eV(result)

    def __next_temperature__(self):
        self.parser.skip("temperature           =")
        return K(self.parser.next_float() * numericalunits.K)

    def temperature(self):
        """
        Retrieves temperature values.

        Returns:

            A numpy array with temperature values across the whole calculation.
        """
        self.parser.reset()
        result = []

        try:
            while True:
                result.append(self.__next_temperature__())
        except StopIteration:
            pass
        return K(result)

    def threads(self):
        """
        Retrieves the number of MPI threads.

        Returns:

            A number of MPI threads for this calculation as an integer.
        """
        return int(re.findall(r"running on\s+(\d+) processors", self.data)[-1])

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
            result.append(self.parser.next_float())

        return numpy.array(result)

    def alat(self):
        """
        Retrieves QE "alat" length.

        Returns:

            The first record of "alat" in **m** as float.
        """
        self.parser.reset()
        self.parser.skip("lattice parameter (alat)")
        alat = self.parser.next_float()

        # Determine alat in more precise way, if possible
        self.parser.skip("celldm(1)=")
        alat_precise = self.parser.next_float()
        if alat_precise != 0:
            return alat_precise * numericalunits.aBohr
        else:
            return alat * numericalunits.aBohr

    def __collect_cell_meta__(self, energy, forces, size, n):
        meta = self.__collect_source_meta__()
        if energy:
            try:
                accuracy = None
                while self.parser.match_closest(("End of self-consistent calculation", "estimated scf accuracy")) == 1:
                    self.parser.skip("estimated scf accuracy")
                    accuracy = self.parser.next_float()
                meta["total-energy"] = self.__next_total__()
                if accuracy is not None:
                    meta["total-energy-error"] = eV(accuracy * numericalunits.Ry)
            except StopIteration:
                pass
        if forces:
            try:
                meta["forces"] = self.__next_forces__()[:size]
            except StopIteration:
                pass
        if n is not None:
            meta["source-index"] = int(n)
        return meta

    @unit_cell
    def cells(self, tag_energy=True, tag_forces=True):
        """
        Retrieves atomic position data.

        Args:
            tag_energy (bool): include total energy values for each unit cell if possible;
            tag_forces (bool): include force values for each unit cell if possible;

        Returns:
            A set of all unit cells found.
        """
        result = []

        alat = self.alat()

        self.parser.reset()

        # Parse initial unit cell
        self.parser.skip("number of atoms/cell")
        n = self.parser.next_int()

        self.parser.skip("crystal axes: (cart. coord. in units of alat)")
        shape = self.parser.next_float((3, 4))[:, 1:] * alat

        self.parser.skip("Cartesian axes")
        coordinates = numpy.zeros((n, 3))
        captions = numpy.zeros(n, dtype=element_type)
        self.parser.next_line(3)

        for i in range(n):
            self.parser.next_int()
            captions[i] = self.parser.next_match(cre_word)
            self.parser.skip("=")
            coordinates[i, :] = self.parser.next_float(3)

        coordinates *= alat
        result.append(CrystalCell.from_cartesian(
            shape, coordinates, captions,
            meta=self.__collect_cell_meta__(tag_energy, tag_forces, len(coordinates), 0)
        ))

        # Parse MD steps
        while True:
            d_apos = self.parser.distance("ATOMIC_POSITIONS", to="tail", default=-1)
            if d_apos == -1:
                # Nothing left
                break

            d_lat = self.parser.distance("CELL_PARAMETERS", to="tail", default=float("inf"))

            # Check if vcr steps are present
            if d_lat < d_apos:

                self.parser.save()
                self.parser.fw(d_lat)
                mode = self.parser.match_closest(("(alat=", "(angstrom)"))
                if mode == 0:
                    alat = self.parser.next_float() * numericalunits.aBohr
                    shape = self.parser.next_float((3, 3)) * alat
                elif mode == 1:
                    shape = self.parser.next_float((3, 3)) * numericalunits.angstrom
                else:
                    raise RuntimeError("Unknown format")
                self.parser.pop()

            # Parse atomic data
            self.parser.fw(d_apos)
            units = self.parser.next_match(cre_word)

            self.parser.next_line()
            coordinates, captions = qe_scf_cell(self.data[self.parser.__position__:], n)
            captions = [bytearray(i).decode() for i in captions]

            meta = self.__collect_cell_meta__(tag_energy, tag_forces, len(coordinates), len(result))
            if units == "crystal":
                result.append(CrystalCell(shape, coordinates, captions, meta=meta))
            elif units == "alat":
                result.append(CrystalCell.from_cartesian(shape, coordinates * alat, captions, meta=meta))
            elif units == "bohr":
                result.append(CrystalCell.from_cartesian(shape, coordinates * numericalunits.aBohr, captions, meta=meta))
            else:
                raise ParseError("Unknown units: %s" % units)

        return result

    def __bands_energies__(self, parseMode_kp, vectors, kpoints, fermi, alat):

        n_kp = len(kpoints)
        energies = []

        for i in range(n_kp):

            self.parser.skip("k =")
            self.parser.next_line(2)
            sub_energies = []

            while self.parser.match_closest((cre_float, cre_word)) == 0:
                sub_energies.append(self.parser.next_float() * numericalunits.eV)

            energies.append(sub_energies)

        meta = self.__collect_source_meta__()
        if parseMode_kp == 0:
            c = BandsPath(vectors, kpoints, energies, fermi=fermi, meta=meta)
        else:
            c = BandsPath.from_cartesian(vectors, kpoints * 2 * math.pi / alat, energies, fermi=fermi, meta=meta)

        return c

    def bands(self, index=-1, skipVCRelaxException=False):
        """
        Retrieves bands.

        Kwargs:

            index (int or None): index of a band structure or ``None``
            if all band structures need to be parsed. Supports negative
            indexing.

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
            raise Exception("Variable cell relaxation output detected. " +
                            "No reciprocal lattice vectors found for relaxed cells. " +
                            "To skip this exception and write kpoints in old basis " +
                            "set skipVCRelaxExeption to True.")

        alat = self.alat()

        self.parser.reset()
        self.parser.skip("reciprocal axes: (cart. coord. in units 2 pi/alat)")
        basis = self.parser.next_float((3, 4))[:, 1:] * 2 * math.pi / alat

        self.parser.skip("number of k points=")
        n_kp = self.parser.next_int()

        if self.parser.present("cryst. coord."):

            parseMode_kp = 0
            self.parser.skip("cryst. coord.")
            kpoints = self.parser.next_float((n_kp, 5))[:, 1:4]

        elif self.parser.present("cart. coord. in units 2pi/alat"):

            parseMode_kp = 1
            self.parser.skip("cart. coord. in units 2pi/alat")
            kpoints = self.parser.next_float((n_kp, 5))[:, 1:4]

        else:
            raise Exception("No kpoint data found in the file.")

        bandStructures = []

        counter = 0

        while True:

            parseMode = self.parser.match_closest(("End of self-consistent calculation", "End of band structure calculation"))

            if parseMode is None:
                break
            elif parseMode == 0:
                self.parser.skip("End of self-consistent calculation")
            elif parseMode == 1:
                self.parser.skip("End of band structure calculation")

            curr_fermi = fermi[counter] if counter < len(fermi) else None

            if index is None:

                bandStructures.append(self.__bands_energies__(parseMode_kp, basis, kpoints, curr_fermi, alat))

            elif index < 0:

                self.parser.save()

            elif index == counter:

                return self.__bands_energies__(parseMode_kp, basis, kpoints, curr_fermi, alat)

            counter += 1

        if index is None:

            return bandStructures

        elif index < 0:

            if (-index <= counter):

                for i in range(-index):
                    self.parser.pop()

                c = self.__bands_energies__(parseMode_kp, basis, kpoints, fermi[index] if -index <= len(fermi) else None,
                                            alat)

                for i in range(counter + index):
                    self.parser.pop()

                return c

        if counter == 0:
            raise ParseError("No band structures found")

        else:
            raise ParseError("Band structure index {:d} is out of range 0-{:d}".format(index, counter-1))

    @band_structure
    def __bands_silent__(self):
        return self.bands(skipVCRelaxException=True)

    def valence(self):
        """
        Retrieves valence electron count per each specimen.

        Returns:
            A dict with electron counts.
        """
        self.parser.reset()
        self.parser.skip("valence")
        self.parser.next_line()
        result = {}
        while self.parser.match_closest(("\n", cre_var_name)) == 1:
            name = self.parser.next_match(cre_var_name)
            value = self.parser.next_float()
            self.parser.next_line()
            result[name] = value

        return result


class Proj(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing output files created by projwfc.x binary of
    Quantum Espresso package.

    Args:

        data (str): string with the contents of the output file.
    """

    @staticmethod
    def valid_header(header):
        return "Program PROJWFC" in header

    @staticmethod
    def valid_filename(name):
        raise NotImplementedError

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

        state = 0

        if self.parser.present("Calling projwave ...."):

            self.parser.skip("Calling projwave ....")

            dataType = numpy.dtype([
                ('state', numpy.int64),
                ('atom', numpy.int64),
                ('name', element_type),
                ('wfc', numpy.int64),
                ('l', numpy.float64),
                ('m', numpy.float64)
            ])

            result = []

            while self.parser.match_closest(("state #", " k = ")) == 0:
                self.parser.skip("state #")
                state += 1
                self.parser.skip(":")

                result.append((
                    state,
                    self.parser.next_int(),
                    self.parser.next_match(cre_word),
                    self.parser.next_int(),
                    self.parser.next_float(),
                    self.parser.next_float()
                ))

            states = numpy.array(result, dtype=dataType)

        elif self.parser.present("Calling projwave_nc ...."):

            self.parser.skip("Calling projwave_nc ....")

            dataType = numpy.dtype([
                ('state', numpy.int64),
                ('atom', numpy.int64),
                ('name', element_type),
                ('wfc', numpy.int64),
                ('j', numpy.float64),
                ('l', numpy.float64),
                ('m_j', numpy.float64)
            ])

            result = []

            while self.parser.match_closest(("state #", " k = ")) == 0:
                self.parser.skip("state #")
                state += 1
                self.parser.skip(":")

                result.append((
                    state,
                    self.parser.next_int(),
                    self.parser.next_match(cre_word),
                    self.parser.next_int(),
                    self.parser.next_float(),
                    self.parser.next_float(),
                    self.parser.next_float()
                ))

            states = numpy.array(result, dtype=dataType)

        else:
            raise Exception("Unknown projwfc output file.")

        return states

    def weights(self):
        """
        Retrieves projection weights onto localized basis set.

        Returns:

            A k by n by m numpy array with weights.

            * k is a number of k points
            * n is a number of bands
            * m is a localized basis set size
        """
        return qe_proj_weights(self.data)

    def _weights(self, lower=0, upper=None):

        basisSize = self.basis().shape[0]

        self.parser.reset()
        self.parser.skip("Calling projwave")

        projections = []

        while self.parser.present("k ="):

            self.parser.skip("k =")
            projections_k = []

            for i in range(lower):
                self.parser.skip("==== e(")

            while self.parser.match_closest(("==== e(", "k =")) == 0:

                self.parser.skip("==== e(")
                self.parser.next_line()
                rawData = self.parser.next_float("|psi|^2")

                projections_ke = numpy.zeros(basisSize)
                for i in range(rawData.shape[0] // 2):
                    projections_ke[int(rawData[2 * i + 1]) - 1] = rawData[2 * i]
                projections_k.append(projections_ke)

                if not upper is None and len(projections_k) >= (upper - lower):
                    break

            projections.append(projections_k)

        return numpy.array(projections)

    def lowdin(self):
        """
        Lowdin occupations.

        Returns:
            A 1D array with Lowdin occupations per each atom.
        """
        self.parser.reset()
        self.parser.skip("Lowdin Charges:")
        i = 1
        result = []
        while True:
            try:
                self.parser.skip(f"Atom #{i: 4d}")
                result.append(self.parser.next_float())
                i += 1
            except StopIteration:
                break
        return numpy.array(result)


class Cond(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing output files created by pwcond.x binary of Quantum
    Espresso package.

    Args:

        data (str): string with the contents of the output file.
    """

    @staticmethod
    def valid_header(header):
        return "Program PWCOND" in header

    @staticmethod
    def valid_filename(name):
        raise NotImplementedError

    def transmission(self, kind="resolved"):
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
                ('energy', numpy.float64),
                ('kx', numpy.float64),
                ('ky', numpy.float64),
                ('incoming', numpy.complex64),
                ('outgoing', numpy.complex64),
                ('transmission', numpy.float64),
            ])
        elif kind == "total":
            dataType = numpy.dtype([
                ('energy', numpy.float64),
                ('kx', numpy.float64),
                ('ky', numpy.float64),
                ('incoming', numpy.complex64),
                ('transmission', numpy.float64),
            ])
        elif kind == "states_in":
            dataType = numpy.dtype([
                ('energy', numpy.float64),
                ('kx', numpy.float64),
                ('ky', numpy.float64),
                ('incoming', numpy.complex64),
            ])
        elif kind == "states_out":
            dataType = numpy.dtype([
                ('energy', numpy.float64),
                ('kx', numpy.float64),
                ('ky', numpy.float64),
                ('outgoing', numpy.complex64),
            ])
        else:
            raise ParseError(
                "Unknown kind: '%s'; should be either 'total', 'resolved', 'states_in' or 'states_out'." % kind)

        result = []
        self.parser.reset()
        previous = numpy.zeros(1, dtype=dataType)

        try:

            while self.parser.present("---  E-Ef"):

                self.parser.skip("---  E-Ef")

                previous['energy'] = self.parser.next_float()
                previous['kx'] = self.parser.next_float()
                previous['ky'] = self.parser.next_float()

                channels = {"left": {"left": None, "right": None},
                            "right": {"left": None, "right": None}}

                for lead in channels.keys():

                    if self.parser.match_closest(("Nchannels of the %s tip =" % lead, "---  E-Ef")) == 0:

                        self.parser.save()
                        self.parser.skip("Nchannels of the %s tip =" % lead)
                        numberOfChannels = self.parser.next_int()

                        for direction in channels[lead].keys():
                            self.parser.save()
                            self.parser.skip("%s moving states:" % direction)
                            self.parser.next_line(2)
                            states = self.parser.next_float((numberOfChannels, 3))
                            channels[lead][direction] = states[:, 0] + 1.j * states[:, 1]
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

                    entry = self.parser.match_closest(("-->", "Total T_j, R_j =", "E-Ef(ev), T ="))
                    while (entry == 0) or (entry == 1):

                        if entry == 0:

                            self.parser.goto("-->")
                            self.parser.rtn()

                            initial = self.parser.next_int() - 1
                            previous["incoming"] = channels["left"]["right"][initial]

                            final = self.parser.next_int() - 1

                            if final < len(channels["right"]["right"]) and kind == "resolved":
                                t = self.parser.next_float()

                                previous["outgoing"] = channels["right"]["right"][final]
                                previous["transmission"] = t
                                result.append(previous)
                                previous = previous.copy()

                            self.parser.next_line()

                        elif entry == 1:

                            self.parser.skip("Total T_j, R_j =")

                            t = self.parser.next_float()
                            self.parser.next_line()

                            if kind == "total":
                                previous["transmission"] = t
                                result.append(previous)
                                previous = previous.copy()

                        entry = self.parser.match_closest(("-->", "Total T_j, R_j =", "E-Ef(ev), T ="))

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


class Input(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing input file for pw.x binary of a Quantum Espresso
    package.

    Args:

        data (str): string with the contents of the input file.
    """

    def __init__(self, file):
        super(Input, self).__init__(file)
        lines = []
        for line in self.data.split('\n'):
            if not line.strip().startswith('!'):
                lines.append(line)
        self.data = "\n".join(lines)
        self.parser = parse(self.data)

    @staticmethod
    def valid_header(header):
        l = header.lower()
        return ("&control" in l or "&system" in l or "&electrons" in l or "&ions" in l) and not "program pwscf" in l

    @staticmethod
    def valid_filename(name):
        raise NotImplementedError

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

            nl = self.parser.next_match(cre_var_name).lower()
            result[nl] = {}

            while True:

                nxt = self.parser.match_closest(("/", cre_var_name))

                if nxt == 0 or nxt == -1:
                    break

                name = self.parser.next_match(cre_var_name).lower()
                self.parser.skip("=")
                dataType = self.parser.match_closest(("false", "true", cre_float, cre_quotedText))
                if dataType == 0:
                    result[nl][name] = False
                    self.parser.skip("false")
                elif dataType == 1:
                    result[nl][name] = True
                    self.parser.skip("true")
                elif dataType == 2:
                    result[nl][name] = self.parser.next_float()
                elif dataType == 3:
                    result[nl][name] = self.parser.next_match(cre_quotedText)[1:-1]
                else:
                    raise Exception("Could not retrieve value for {}.{}".format(nl, name))

            self.parser.skip("/")

        return result

    @unit_cell
    def cell(self):
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

            shape = list(nl["system"]["celldm(%i)" % (i + 1)] for i in range(6))
            shape[0] *= numericalunits.aBohr
            shape[1] *= shape[0]
            shape[2] *= shape[0]
            basis = RealSpaceBasis.triclinic(shape[:3], shape[3:])

        elif ibrav == 0:

            self.parser.reset()
            self.parser.skip("cell_parameters")
            units = self.parser.next_match(cre_word)
            vectors = self.parser.next_float(n=(3, 3)) * units_dict[units]
            basis = RealSpaceBasis(vectors)

        elif ibrav == 2:

            a = nl["system"]["celldm(1)"]
            basis = RealSpaceBasis(a / 2 * numericalunits.aBohr * numpy.array((
                (-1, 0, 1), (0, 1, 1), (-1, 1, 0),
            )))

        else:
            raise NotImplementedError("Cell recovery not implemented for ibrav = {:d}".format(int(ibrav)))

        self.parser.reset()
        self.parser.skip("atomic_positions")
        units = self.parser.next_match(cre_word, n="\n")
        if len(units) == 0:
            units = "alat"
        else:
            units = units[0]
        coordinates = numpy.zeros((int(nl["system"]["nat"]), 3))
        statics = numpy.ones(coordinates.shape, dtype=numpy.float)
        values = numpy.zeros(coordinates.shape[0], dtype=element_type)

        for i in range(coordinates.shape[0]):

            values[i] = self.parser.next_match(cre_word)
            coordinates[i, :] = self.parser.next_float(3)

            if self.parser.match_closest(("\n", cre_int)) == 1:
                statics[i, :] = self.parser.next_float(3)

            self.parser.next_line()

        if units == "alat":
            result = CrystalCell.from_cartesian(basis, coordinates * units_dict["alat"], values)
        elif units == "bohr":
            result = CrystalCell.from_cartesian(basis, coordinates * numericalunits.aBohr, values)
        elif units == "angstrom":
            result = CrystalCell.from_cartesian(basis, coordinates * numericalunits.angstrom, values)
        elif units == "crystal":
            result = CrystalCell(basis, coordinates, values)
        else:
            raise RuntimeError("Unknown units: {}".format(units))

        result.meta["force-factors"] = statics

        return result


# Lower case versions
bands = Bands
output = Output
cond = Cond
proj = Proj
input = Input
