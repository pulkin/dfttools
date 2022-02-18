"""
Parsing various atomic structure files.
"""
import numericalunits
import numpy

from .generic import cre_word, cre_non_space, AbstractTextParser, IdentifiableParser
from ..data import element_for_number
from ..simple import unit_cell
from ..types import CrystalCell, CrystalGrid, RealSpaceBasis, element_type


class XSF(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing `xcrysden <http://www.xcrysden.org>`_ files
    commonly used in solid state visualisations.
    """

    @staticmethod
    def valid_filename(name):
        return name.lower().endswith(".xsf")

    @staticmethod
    def valid_header(header):
        l = header.lower()
        return "primvec" in l and "primcoord" in l

    @unit_cell
    def cells(self):
        """
        Retrieves unit cells.
        
        Returns:
        
            A set of unit cells with atomic positions data.
        """
        result = []

        self.parser.reset()

        while True:

            mode = self.parser.match_closest(("primvec", "primcoord"))

            if mode == 0:
                self.parser.skip("primvec")
                self.parser.next_line()
                shape = self.parser.next_float((3, 3)) * numericalunits.angstrom

            elif mode == 1:
                self.parser.skip('primcoord')
                self.parser.next_line()
                n = self.parser.next_int()
                coordinates = numpy.zeros((n, 3))
                values = numpy.zeros(n, dtype=element_type)
                for i in range(n):
                    self.parser.next_line()
                    values[i] = self.parser.next_match(cre_word)
                    coordinates[i, :] = self.parser.next_float(3) * numericalunits.angstrom
                result.append(CrystalCell.from_cartesian(
                    shape,
                    coordinates,
                    values,
                ))

            else:
                return result

    def grids(self):
        """
        Retrieves the grids.
        
        Returns:
        
            An array of cells with data on the grid. ( grid origin, grid vectors, data on the
            grid, grid name, grid block name ).
        """
        result = []

        self.parser.reset()

        while self.parser.present("begin_block_datagrid"):

            self.parser.skip("begin_block_datagrid")

            if self.parser.distance("_3d", default=-1) == 0:
                mode = "3d"
            elif self.parser.distance("_2d", default=-1) == 0:
                mode = "2d"
            else:
                raise Exception("Failed to determine grid dimensions")

            self.parser.next_line()
            block_name = self.parser.next_match(cre_non_space)

            while True:

                self.parser.next_line()
                grid_name = self.parser.next_match(cre_non_space)
                expecting = "begin_datagrid_" + mode
                expecting_2 = "datagrid_" + mode

                if grid_name.lower() == "end_block_datagrid_" + mode:
                    break
                elif grid_name.lower().startswith(expecting):
                    grid_name = grid_name[len(expecting) + 1:]
                elif grid_name.lower().startswith(expecting_2):
                    grid_name = grid_name[len(expecting_2) + 1:]
                else:
                    raise Exception("Failed to continue parsing grids at '{}'".format(grid_name))

                if mode == "3d":
                    shape = self.parser.next_int(3)
                else:
                    shape = self.parser.next_int(2)

                origin = self.parser.next_float(3) * numericalunits.angstrom

                if mode == "3d":
                    vectors = self.parser.next_float((3, 3)) * numericalunits.angstrom
                else:
                    vectors = self.parser.next_float((2, 3)) * numericalunits.angstrom

                data = self.parser.next_float(tuple(shape[::-1])).swapaxes(0, shape.size - 1)[
                    (slice(0, -1, 1),) * vectors.shape[0]]

                self.parser.skip("end_datagrid_" + mode)

                meta = {
                    "xsf-grid-origin": origin,
                    "xsf-block-name": block_name,
                    "xsf-grid-name": grid_name,
                }
                c = CrystalGrid(
                    vectors,
                    tuple(numpy.linspace(0, 1, s, endpoint=False) for s in data.shape),
                    data,
                    meta=meta,
                )
                result.append(c)

        return result


class GaussianCube(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing `Gaussian CUBE <http://www.gaussian.com/>`_ files.
    """

    @staticmethod
    def valid_header(header):
        raise NotImplementedError

    @staticmethod
    def valid_filename(name):
        return name.lower().endswith(".cube")

    def grid(self):
        """
        Retrieves the grid.
        
        Returns:
        
            A grid.
        """
        self.parser.reset()
        self.parser.next_line(2)

        n = self.parser.next_int()
        if n < 0:
            origin = self.parser.next_float(3) * numericalunits.angstrom
        else:
            origin = self.parser.next_float(3) * numericalunits.aBohr
        if not numpy.all(origin == 0):
            raise NotImplementedError("Ambiguous origin shift in CUBE is not implemented")
        n = abs(n)

        size = []
        spacing = []

        for i in range(3):
            s = self.parser.next_int()
            size.append(s)

            if s < 0:
                spacing.append(self.parser.next_float(3) * numericalunits.angstrom)
            else:
                spacing.append(self.parser.next_float(3) * numericalunits.aBohr)

        spacing = numpy.array(spacing)
        size = numpy.abs(size)
        vectors = spacing * size[:, numpy.newaxis]

        # Skip atomic coordinates
        self.parser.next_line(n + 1)

        data = self.parser.next_float(size)

        return CrystalGrid(
            vectors,
            tuple(numpy.linspace(0, 1, s, endpoint=False) for s in data.shape),
            data,
        )

    @unit_cell
    def cell(self):
        """
        Retrieves a unit cell.
        
        Returns:
        
            A unit cell with atomic positions data.
        """
        result = []

        self.parser.reset()
        self.parser.next_line(2)

        # Number of atoms
        n = self.parser.next_int()
        self.parser.next_line()

        # Lattice vectors
        shape = []
        for i in range(3):

            nv = self.parser.next_int()
            v = self.parser.next_float(3)

            if nv < 0:
                shape.append(v * abs(nv) * numericalunits.angstrom)
            else:
                shape.append(v * abs(nv) * numericalunits.aBohr)

        c = []
        v = []

        for i in range(abs(n)):

            aid = self.parser.next_int()
            self.parser.next_float()
            ac = self.parser.next_float(3)

            if not aid == 0:
                v.append(element_for_number[abs(aid)])
            else:
                v.append("??")

            if nv < 0:
                c.append(ac * numericalunits.angstrom)
            else:
                c.append(ac * numericalunits.aBohr)

        return CrystalCell.from_cartesian(shape, c, v)


class XYZ(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing XYZ structure files.
    """

    vacuum_size = numericalunits.nm

    @staticmethod
    def valid_header(header):
        raise NotImplementedError

    @staticmethod
    def valid_filename(name):
        return name.lower().endswith(".xyz")

    @unit_cell
    def cell(self):
        """
        Retrieves a unit cell.
        
        Returns:
        
            A unit cell with atomic positions data.
        """
        self.parser.reset()

        # Number of atoms
        n = self.parser.next_int()
        self.parser.next_line(2)

        c = []
        v = []

        for i in range(abs(n)):
            v.append(self.parser.next_match(cre_word))
            c.append(self.parser.next_float(3) * numericalunits.angstrom)

        c = numpy.array(c)
        mx = c.max(axis=0)
        mn = c.min(axis=0)
        shape = mx - mn + XYZ.vacuum_size

        return CrystalCell.from_cartesian(RealSpaceBasis.orthorhombic(shape), c, v)


class CIF(AbstractTextParser, IdentifiableParser):
    """
    Class for parsing CIF files.
    """
    @staticmethod
    def valid_filename(name):
        return ".cif" in name

    @staticmethod
    def valid_header(header):
        return "loop_" in header

    def basis(self):
        """
        Retrieves the crystal basis.

        Returns:
            The crystal basis.
        """
        self.parser.reset()
        tokens = ("_cell_length_a", "_cell_length_b", "_cell_length_c",
                  "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")
        vecs = [None] * 6

        while True:
            c = self.parser.match_closest(tokens)
            if c is None:
                break
            self.parser.skip(tokens[c])
            x = self.parser.next_float()
            if c < 3:
                vecs[c] = x * numericalunits.angstrom
            else:
                vecs[c] = numpy.cos(x * numpy.pi / 180)
        if any(i is None for i in vecs):
            raise ValueError("Missing the following tokens: {}".format(", ".join(
                t for i, t in zip(vecs, tokens) if i is None
            )))
        return RealSpaceBasis.triclinic(vecs[:3], vecs[3:])

    @unit_cell
    def cells(self):
        """
        Retrieves the unit cell.

        Returns:
            The unit cell with atomic positions data.
        """
        basis = self.basis()
        self.parser.reset()

        tokens = "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_label", \
                 "_atom_site_type_symbol"
        cells = []
        while self.parser.present("loop_"):
            while True:
                ptrs = [None] * 5
                ptr_set = [False] * 5
                self.parser.skip("loop_")
                fields = []
                self.parser.next_line()
                while self.parser.distance("_", default=-1) == 0:
                    fields.append(self.parser.next_match(cre_non_space))
                    self.parser.next_line()

                for i, t in enumerate(tokens):
                    if t in fields:
                        ptrs[i] = fields.index(t)
                        ptr_set[i] = True

                if all(ptr_set[:3]) and any(ptr_set[3:]):
                    break

            data = []
            while self.parser.match_closest(("_", cre_non_space)) == 1:
                data.append(self.parser.next_match(cre_non_space, len(fields)))

            data = numpy.array(data)
            coords = data[:, ptrs[:3]].astype(float)
            if ptr_set[4]:
                vals = map(str.lower, data[:, ptrs[4]])
            else:
                vals = data[:, ptrs[3]].astype(str)
            cells.append(CrystalCell(basis, coords, tuple(vals)))
        return cells


# Lower case versions
xsf = XSF
cube = GaussianCube
xyz = XYZ
cif = CIF