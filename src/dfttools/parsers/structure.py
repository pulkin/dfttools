"""
Parsing various atomic structure files.
"""
import numericalunits
import numpy

from . import default_real_space_basis
from .generic import cre_word, cre_nonspace, AbstractParser
from ..presentation import __elements_table__
from ..simple import unit_cell
from ..types import UnitCell, Grid


class XSF(AbstractParser):
    """
    Class for parsing `xcrysden <http://www.xcrysden.org>`_ files
    commonly used in solid state visualisations.
    
    Args:
    
        data (str): string with the contents of the xsf file.
    """

    @staticmethod
    def valid_filename(name):
        return name.lower().endswith(".xsf")

    @staticmethod
    def valid_header(header):
        l = header.lower()
        return "primvec" in l and "primcoord" in l

    @unit_cell
    def unitCells(self):
        """
        Retrieves unit cells.
        
        Returns:
        
            A set of unit cells with atomic positions data.
        """
        result = []

        self.parser.reset()

        while True:

            mode = self.parser.closest(("primvec", "primcoord"))

            if mode == 0:
                self.parser.skip("primvec")
                self.parser.nextLine()
                shape = self.parser.nextFloat((3, 3)) * numericalunits.angstrom

            elif mode == 1:
                self.parser.skip('primcoord')
                self.parser.nextLine()
                n = self.parser.nextInt()
                coordinates = numpy.zeros((n, 3))
                values = numpy.zeros(n, dtype='S2')
                for i in range(n):
                    self.parser.nextLine()
                    values[i] = self.parser.nextMatch(cre_word)
                    coordinates[i, :] = self.parser.nextFloat(3) * numericalunits.angstrom
                result.append(UnitCell(
                    default_real_space_basis(shape),
                    coordinates,
                    values,
                    c_basis='cartesian'
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

            self.parser.nextLine()
            block_name = self.parser.nextMatch(cre_nonspace)

            while True:

                self.parser.nextLine()
                grid_name = self.parser.nextMatch(cre_nonspace)
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
                    shape = self.parser.nextInt(3)
                else:
                    shape = self.parser.nextInt(2)

                origin = self.parser.nextFloat(3) * numericalunits.angstrom

                if mode == "3d":
                    vectors = self.parser.nextFloat((3, 3)) * numericalunits.angstrom
                else:
                    vectors = self.parser.nextFloat((2, 3)) * numericalunits.angstrom

                data = self.parser.nextFloat(tuple(shape[::-1])).swapaxes(0, shape.size - 1)[
                    [slice(0, -1, 1)] * vectors.shape[0]]

                self.parser.skip("end_datagrid_" + mode)

                meta = {
                    "xsf-grid-origin": origin,
                    "xsf-block-name": block_name,
                    "xsf-grid-name": grid_name,
                }
                c = Grid(
                    default_real_space_basis(vectors, meta=meta),
                    tuple(numpy.linspace(0, 1, s, endpoint=False) for s in data.shape),
                    data,
                )
                result.append(c)

        return result


class GaussianCube(AbstractParser):
    """
    Class for parsing `Gaussian CUBE <http://www.gaussian.com/>`_ files.
    
    Args:
    
        data (str): string with the contents of the Gaussian CUBE file.
    """

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
        self.parser.nextLine(2)

        n = self.parser.nextInt()
        if n < 0:
            origin = self.parser.nextFloat(3) * numericalunits.angstrom
        else:
            origin = self.parser.nextFloat(3) * numericalunits.aBohr
        n = abs(n)

        size = []
        spacing = []

        for i in range(3):
            s = self.parser.nextInt()
            size.append(s)

            if s < 0:
                spacing.append(self.parser.nextFloat(3) * numericalunits.angstrom)
            else:
                spacing.append(self.parser.nextFloat(3) * numericalunits.aBohr)

        spacing = numpy.array(spacing)
        size = numpy.abs(size)
        vectors = spacing * size[:, numpy.newaxis]

        # Skip atomic coordinates
        self.parser.nextLine(n + 1)

        data = self.parser.nextFloat(size)

        return Grid(
            default_real_space_basis(vectors),
            tuple(numpy.linspace(0, 1, s, endpoint=False) for s in data.shape),
            data,
        )

    @unit_cell
    def unitCell(self):
        """
        Retrieves a unit cell.
        
        Returns:
        
            A unit cell with atomic positions data.
        """
        result = []

        self.parser.reset()
        self.parser.nextLine(2)

        # Number of atoms
        n = self.parser.nextInt()
        self.parser.nextLine()

        # Lattice vectors
        shape = []
        for i in range(3):

            nv = self.parser.nextInt()
            v = self.parser.nextFloat(3)

            if nv < 0:
                shape.append(v * abs(nv) * numericalunits.angstrom)
            else:
                shape.append(v * abs(nv) * numericalunits.aBohr)

        c = []
        v = []

        for i in range(abs(n)):

            aid = self.parser.nextInt()
            self.parser.nextFloat()
            ac = self.parser.nextFloat(3)

            if not aid == 0:
                v.append(__elements_table__[abs(aid) - 1][0])
            else:
                v.append("??")

            if nv < 0:
                c.append(ac * numericalunits.angstrom)
            else:
                c.append(ac * numericalunits.aBohr)

        return UnitCell(default_real_space_basis(shape), c, v, c_basis="cartesian")


class XYZ(AbstractParser):
    """
    Class for parsing XYZ structure files.
    
    Args:
    
        data (str): string with the contents of the XYZ file.
    """

    vacuum_size = numericalunits.nm

    @staticmethod
    def valid_filename(name):
        return name.lower().endswith(".xyz")

    @unit_cell
    def unitCell(self):
        """
        Retrieves a unit cell.
        
        Returns:
        
            A unit cell with atomic positions data.
        """
        result = []

        self.parser.reset()

        # Number of atoms
        n = self.parser.nextInt()
        self.parser.nextLine(2)

        c = []
        v = []

        for i in range(abs(n)):
            v.append(self.parser.nextMatch(cre_word))
            c.append(self.parser.nextFloat(3) * numericalunits.angstrom)

        c = numpy.array(c)
        mx = c.max(axis=0)
        mn = c.min(axis=0)
        shape = mx - mn + XYZ.vacuum_size

        return UnitCell(default_real_space_basis(shape, kind='orthorombic'), c, v, c_basis="cartesian")


# Lower case versions
xsf = XSF
cube = GaussianCube
xyz = XYZ
