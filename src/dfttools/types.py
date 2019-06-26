"""
This submodule contains key types for handling coordinate-dependent data:
``UnitCell``, ``Grid`` and ``Basis``.
"""
import itertools
from functools import wraps

import numpy
from numpy import random
from scipy.spatial import cKDTree

from .blochl import tetrahedron, tetrahedron_plain
from .util import cast_units


def input_as_list(func):
    @wraps(func)
    def a_w(*args, **kwargs):
        self = args[0]
        if len(args) > 2:
            args = [self, list(args[1:])]
        elif len(args) == 2:
            try:
                iter(args[1])
            except TypeError:
                args = [self, list(args[1:])]
        else:
            args = [self, []]

        return func(*args, **kwargs)

    return a_w


def __angle__(v1, v2, axis=-1):
    """
    Calculates angles between sets of vectors.
    
    Args:
    
        v1,v2 (array): arrays of the same size with vectors'
        coordinates.
        
    Kwargs:
    
        axis (int): dimension to sum over.
        
    Returns:
    
        A numpy array containing cosines between the vectors.
    """
    return (v1 * v2).sum(axis=axis) / ((v1 ** 2).sum(axis=axis) * (v2 ** 2).sum(axis=axis)) ** .5


def __xyz2i__(i):
    try:
        return {'x': 0, 'y': 1, 'z': 2}[i]
    except KeyError:
        return i


class ArgumentError(Exception):
    pass


class Basis(object):
    """
    A class describing a set of vectors representing a basis.
    
    Args:
    
        vectors (array): a 2D or a 1D array of floats representing
        vectors of the basis set.
        
    Kwargs:
    
        kind (str): a shortcut keyword for several most common basis
        sets:
        
        * 'default': expects ``vectors`` to be a 2D array with basis
          vectors in cartesian coordinates;
        * 'orthorombic': expects ``vectors`` to be a 1D array with
          dimensions of an orthorombic basis set;
        * 'triclinic': expects ``vectors`` to be a 1D array with 3
          lengths of edges and 3 cosines of face angles.

        meta (dict): a metadata for this Basis.
    """

    def __init__(self, vectors, kind='default', meta=None):

        if isinstance(vectors, Basis):
            self.vectors = numpy.asanyarray(vectors.vectors)
            if meta is None:
                meta = vectors.meta.copy()
            else:
                meta = meta.copy()
                meta.update(vectors.meta)

        else:
            vectors = numpy.asanyarray(vectors, dtype=numpy.float64)

            if kind == 'default':
                self.vectors = vectors

            elif kind == 'orthorombic':
                self.vectors = cast_units(numpy.diag(vectors), vectors)

            elif kind == 'triclinic':
                lengths = vectors[0:3]
                cosines = vectors[3:]
                volume = lengths[0] * lengths[1] * lengths[2] * (
                        1 + \
                        2 * cosines[0] * cosines[1] * cosines[2] - \
                        cosines[0] ** 2 - cosines[1] ** 2 - cosines[2] ** 2
                ) ** .5
                sines = (1 - cosines ** 2) ** .5
                height = volume / lengths[0] / lengths[1] / sines[2]
                self.vectors = cast_units(numpy.asanyarray((
                    (lengths[0], 0, 0),
                    (lengths[1] * cosines[2], lengths[1] * sines[2], 0),
                    (lengths[2] * cosines[1], abs((lengths[2] * sines[1]) ** 2 - height ** 2) ** .5, height)
                ), dtype=numpy.float64), vectors)

            else:
                raise ArgumentError("Unknown kind='{}'".format(kind))

        if meta is not None:
            self.meta = meta.copy()
        else:
            self.meta = {}

    def __getstate__(self):
        return dict(
            vectors=self.vectors.copy(),
            meta=self.meta.copy(),
        )

    def __setstate__(self, data):
        Basis.__init__(
            self,
            data["vectors"],
            meta=data["meta"],
        )

    def __eq__(self, another):
        return type(another) == type(self) and numpy.all(self.vectors == another.vectors)

    @classmethod
    def class_id(cls):
        """
        Retrieves a unique ID of the class.
        Returns:
            A string ID of the class based on class and module names.
        """
        return cls.__module__ + "." + getattr(cls, "__qualname__", cls.__name__)

    def to_json(self):
        """
        Prepares a JSON-compatible object representing this Basis.
        
        Returns:
        
            A JSON-compatible dict.
        """
        result = self.__getstate__()
        result["type"] = self.class_id()
        return result

    @classmethod
    def from_json(cls, j):
        """
        Restores a Basis from JSON data.
        
        Args:
        
            j (dict): JSON data.
            
        Returns:
        
            A Basis object.
        """
        j = dict(j)
        if "type" not in j or j["type"] != cls.class_id():
            raise ValueError("Invalid JSON, expected type {}".format(cls.class_id()))
        del j["type"]
        result = cls(**j)
        return result

    def transform_to(self, basis, coordinates):
        """
        Transforms coordinates to another basis set.
        
        Args:
        
            basis (Basis): a new basis to transform to.
            
            coordinates (array): an array of coordinates to be
            transformed.
            
        Returns:
        
            An array with transformed coordinates.
        """
        coordinates = numpy.asanyarray(coordinates, dtype=numpy.float64)
        return numpy.tensordot(
            coordinates,
            numpy.tensordot(
                self.vectors,
                numpy.linalg.inv(basis.vectors),
                axes=((1,), (0,))
            ),
            axes=((len(coordinates.shape) - 1,), (0,))
        )

    def transform_from(self, basis, coordinates):
        """
        Transforms coordinates from another basis set.
        
        Args:
        
            basis (Basis): a basis to transform from.
            
            coordinates (array): an array of coordinates to be
            transformed.
            
        Returns:
        
            An array with transformed coordinates.
        """
        return basis.transform_to(self, coordinates)

    def transform_to_cartesian(self, coordinates):
        """
        Transforms coordinates to cartesian.
        
        Args:
            
            coordinates (array): an array of coordinates to be
            transformed.
            
        Returns:
        
            An array with transformed coordinates.
        """
        return self.transform_to(
            Basis(numpy.eye(self.vectors.shape[0])),
            coordinates,
        )

    def transform_from_cartesian(self, coordinates):
        """
        Transforms coordinates from cartesian.
        
        Args:
            
            coordinates (array): an array of coordinates to be
            transformed.
            
        Returns:
        
            An array with transformed coordinates.
        """
        return self.transform_from(
            Basis(numpy.eye(self.vectors.shape[0])),
            coordinates,
        )

    def rotated(self, axis, angle, units='rad'):
        """
        Rotates this basis.
        
        Args:
        
            axis (array): axis to rotate around;
            
            angle (float): angle to rotate;
            
        Kwargs:
        
            units (str): units of the angle: 'rad', 'deg' or 'frac'.
            
        Returns:
        
            A rotated copy of this basis.
        """
        units = {
            "rad": 1.0,
            "deg": numpy.pi / 180,
            "frac": numpy.pi * 2,
        }[units]
        angle *= units
        c = numpy.cos(angle)
        s = numpy.sin(angle)
        axis = numpy.asanyarray(axis, dtype=numpy.float)
        axis /= (axis ** 2).sum() ** .5
        axis_x = numpy.asanyarray((
            (0, -axis[2], axis[1]),
            (axis[2], 0, -axis[0]),
            (-axis[1], axis[0], 0),
        ))
        M = c * numpy.eye(self.vectors.shape[0]) + s * axis_x + (1 - c) * numpy.dot(axis[:, numpy.newaxis],
                                                                                    axis[numpy.newaxis, :])
        result = self.copy()
        result.vectors = numpy.dot(result.vectors, M)
        return result

    def volume(self):
        """
        Computes the volume of a triclinic cell represented by the basis.
        
        Returns:
        
            Volume of the cell in **m^3**.
        """
        return abs(numpy.linalg.det(self.vectors))

    def reciprocal(self):
        """
        Computes a reciprocal basis.
        
        Returns:
        
            A reciprocal basis.
            
        .. note::
        
            The :math:`2 \pi` multiplier is not present.
        
        """
        return Basis(cast_units(numpy.swapaxes(numpy.linalg.inv(self.vectors), 0, 1), self.vectors, inv=True))

    def vertices(self):
        """
        Computes cartesian coordinates of all vertices of the basis cell.
        
        Returns:
        
            Cartesian coordinates of all vertices.
        """
        result = []
        for v in itertools.product((0.0, 1.0), repeat=self.vectors.shape[0]):
            result.append(self.transform_to_cartesian(numpy.asanyarray(v)))
        return numpy.asanyarray(result)

    def edges(self):
        """
        Computes pairs of cartesian coordinates of all edges of the
        basis cell.
        
        Returns:
        
            A list of pairs with cartesian coordinates of vertices
            forming edges.
        """
        result = []
        for e in range(self.vectors.shape[0]):
            for v in itertools.product((0.0, 1.0), repeat=self.vectors.shape[0] - 1):
                v1 = v[:e] + (0.,) + v[e:]
                v2 = v[:e] + (1.,) + v[e:]
                result.append((
                    (self.vectors * numpy.asanyarray(v1)[:, numpy.newaxis]).sum(axis=0),
                    (self.vectors * numpy.asanyarray(v2)[:, numpy.newaxis]).sum(axis=0),
                ))
        return numpy.asanyarray(result)

    def faces(self):
        """
        Computes faces and returns corresponding cartesian coordinates.
        
        Returns:
        
            A list of lists of coordinates defining face polygon coordinates.
        """
        raise NotImplementedError

    def copy(self):
        """
        Calculates a copy.
        
        Returns:
        
            A deep copy of self.
        """
        return self.from_json(self.to_json())

    @input_as_list
    def stack(self, basises, vector='x', tolerance=1e-10, restrict_collinear=False):
        """
        Stacks several basises along one of the vectors.
        
        Args:
        
            basises (list): basises to stack. Corresponding
            vectors of all basises being stacked should match.
            
        Kwargs:
        
            vector (str,int): a vector along which to stack, either 'x',
            'y', 'z' or an int specifying the vector;
        
            tolerance (float): a largest possible error in input basises'
            vectors;
            
            restrict_collinear (bool): if True will raise an exception
            if the vectors to stack along are not collinear
            
        Raises:
        
            ArgumentError: in the case of vector mismatch.
            
        Returns:
        
            A larger basis containing all argument cell stacked.
        """
        basises = [self] + basises
        d = __xyz2i__(vector)

        otherVectors = list(range(basises[0].vectors.shape[0]))
        del otherVectors[d]

        # 3d array with lattice vectors: shapes[i,j,k] i=cell, j=lattice vector, k = component
        shapes = numpy.concatenate(tuple(
            i.vectors[numpy.newaxis, ...] for i in basises
        ), axis=0)

        # Check if non-stacking lattice vectors coincide
        stackingVectorsSum = shapes[:, d, :].sum(axis=0)
        vecLengths = (shapes ** 2).sum(axis=2) ** 0.5
        otherVectors_d = shapes[:, otherVectors, :] - shapes[0, otherVectors, :][numpy.newaxis, ...]
        otherVectors_ds = (otherVectors_d ** 2).sum(axis=-1) ** .5

        if numpy.any(otherVectors_ds > tolerance * vecLengths[:, otherVectors]):
            raise ArgumentError(
                'Dimension mismatch for stacking:\n{}\nCheck your input basis vectors or set tolerance to at least {} to skip this error'.format(
                    shapes,
                    numpy.amax(otherVectors_ds / vecLengths[:, otherVectors]),
                ))

        if restrict_collinear and numpy.any(
                numpy.abs(__angle__(shapes[:, d, :], stackingVectorsSum[numpy.newaxis, :]) - 1) > tolerance):
            raise ArgumentError(
                'Vectors to stack along are not collinear:\n{}\nCheck your input basis vectors or set tolerance to at least {} to skip this error'.format(
                    shapes,
                    numpy.amax(numpy.abs(__angle__(shapes[:, d, :], stackingVectorsSum[numpy.newaxis, :]) - 1)),
                ))

        shape = self.vectors.copy()
        shape[d, :] = stackingVectorsSum

        return Basis(shape, meta=self.meta)

    @input_as_list
    def repeated(self, times):
        """
        Produces a new basis from a given by repeating it in all
        directions.
        
        Args:
        
            times (array): array of ints specifying how much the basis
            should be repeated along each of the vectors.
        """
        c = self
        for i, t in enumerate(times):
            if not isinstance(t, int):
                raise ValueError("The input [{:d}] should be integers, found {} instead".format(i, times))
            c = c.stack(*((c,) * (t - 1)), vector=i)

        return c

    @input_as_list
    def reorder_vectors(self, new):
        """
        Reorders vectors.
        
        Args:
        
            new (array): new mapping of vectors.
            
        Example:
        
            >>> basis.reorder_vectors(0, 1, 2) # does nothing
            >>> basis.reorder_vectors(1, 0, 2) # swaps first and second vectors.
        """
        new = tuple(__xyz2i__(i) for i in new)

        if not len(new) == self.vectors.shape[0]:
            raise ArgumentError(
                "The new mapping of vectors should be of the size of {:d}".format(self.vectors.shape[0]))

        if not len(set(new)) == self.vectors.shape[0]:
            raise ArgumentError("The input contains duplicates")

        self.vectors = self.vectors[new, :]

    def generate_path(self, points, n, anchor=True):
        """
        Generates a path given key points and the total number of points
        on the path.
                
        Args:
        
            points (array): key points of the path expressed in this basis;
            
            n (int): the total number of points on the path.
            
        Kwargs:
        
            anchor (bool): force the specified points to be present in
            the final path. If True alters slightly the total point number;
            
        Returns:
        
            Path coordinates expressed in this basis.
        """
        points = numpy.asanyarray(points)
        points_c = self.transform_to_cartesian(points)
        lengths = ((points_c[1:] - points_c[:-1]) ** 2).sum(axis=-1) ** .5
        lengths_cs = numpy.cumsum(lengths)

        if anchor:

            lengths_n = numpy.round(lengths * n / lengths_cs[-1]).astype(numpy.int)
            path = []

            for i in range(lengths_n.size):
                path_pos = numpy.linspace(0, 1, lengths_n[i], endpoint=False)[:, numpy.newaxis]
                path.append(points[i, numpy.newaxis, :] * (1 - path_pos) + points[i + 1, numpy.newaxis, :] * path_pos)

            path.append(points[-1, numpy.newaxis, :])
            path = numpy.concatenate(path, axis=0)

        else:

            path_l = numpy.linspace(0, 1, n) * lengths_cs[-1]
            path_id = numpy.searchsorted(lengths_cs, path_l)
            path_pos = (lengths_cs[path_id] - path_l) / lengths[path_id]
            path = points[path_id, :] * path_pos[:, numpy.newaxis] + points[path_id + 1, :] * (
                        1 - path_pos[:, numpy.newaxis])

        return numpy.asanyarray(path)


def diamond_basis(a):
    """
    Creates a diamond basis with a given lattice constant.
    
    Args:
    
        a (float): the lattice constant;
        
    Returns:
    
        A diamond Basis.
    """
    a = 0.5 * a
    return Basis([[0, a, a], [a, 0, a], [a, a, 0]])


class UnitCell(Basis):
    """
    A class describing a crystal unit cell in a periodic environment.
    
    Args:
    
        vectors (Basis,array): a crystal basis.
        
        coordinates (array): a 2D array of coordinates of atoms (or any
        other instances)
        
        values (array): an array of atoms (or any other instances) with
        the leading dimenstion being the same as the one of
        ``coordinates`` array.

    Kwargs:

        meta (dict): a metadata for this UnitCell;

        c_basis (str,Basis): a Basis for input coordinates or 'cartesian'
        if coordinates are passed in the cartesian basis.
    """

    def __init__(self, vectors, coordinates, values, meta=None, c_basis=None):

        if isinstance(vectors, Basis):
            Basis.__init__(self, vectors.vectors, meta=vectors.meta)
        else:
            Basis.__init__(self, vectors)
        if meta is not None:
            self.meta.update(meta)

        dims = self.vectors.shape[0]

        # Process coordinates and vectors input
        self.coordinates = numpy.asanyarray(coordinates, dtype=numpy.float64)

        if len(self.coordinates.shape) == 1:
            if not self.coordinates.shape == (dims,):
                raise ArgumentError(
                    'Coordinates array is 1D, {:d} coordinates have to be specified instead of {:d}'.format(
                        dims,
                        self.coordinates.shape[0],
                    ))

            self.coordinates = self.coordinates[numpy.newaxis, ...]

        elif len(self.coordinates.shape) == 2:
            if self.coordinates.shape[1] != dims:
                raise ArgumentError(
                    'Coordinates array is 2D but the last dimension {:d} is not equal to space dimensionality {:d}'.format(
                        self.coordinates.shape[1], dims))

        # Coordinates are now prepeared, proceed to values
        self.values = numpy.asanyarray(values)
        if len(self.values.shape) == 0:
            self.values = self.values[numpy.newaxis, ...]

        if self.values.shape[0] < self.coordinates.shape[0] and self.coordinates.shape[0] % self.values.shape[0] == 0:
            # Broadcast values repeatedly
            nrep = len(self.coordinates) // len(self.values) + 1
            self.values = numpy.tile(self.values, (nrep,) + (1,) * (self.values.ndim - 1))[:len(self.coordinates)]

        elif not self.values.shape[0] == self.coordinates.shape[0]:
            raise ArgumentError('Mismatch of sizes of coordinates and values arrays: {:d} vs {:d}'.format(
                self.coordinates.shape[0],
                self.values.shape[0]
            ))

        # Process basis information
        if c_basis == None:
            pass

        elif c_basis == 'cartesian':
            self.coordinates = self.transform_from_cartesian(self.coordinates)

        else:
            self.coordinates = self.transform_from(c_basis, self.coordinates)

    def __getstate__(self):
        result = super(UnitCell, self).__getstate__()
        result.update(dict(
            coordinates=self.coordinates.copy(),
            values=self.values.copy(),
        ))
        return result

    def __setstate__(self, data):
        super(UnitCell, self).__setstate__(data)
        self.__init__(self, data["coordinates"], data["values"])

    def __eq__(self, another):
        return Basis.__eq__(self, another) and numpy.all(self.coordinates == another.coordinates) and numpy.all(
            self.values == another.values)

    @input_as_list
    def angles(self, ids):
        """
        Computes angles between cell specimens.
        
        Args:
        
            ids (array): a set of specimen IDs to compute angles between.
            Several shapes are accepted:
            
            * nx3 array: computes n cosines of angles [n,0]-[n,1]-[n,2];
            * 1D array of length n: computes n-2 cosines of angles
              [n-1]-[n]-[n+1];
            
        Returns:
        
            A numpy array containing cosines of angles specified.
            
        Example:
        
            Following are the valid calls:
            
            >>> cell.angles((0,1,2)) # angle between vectors connecting {second and first} and {second and third} species
            >>> cell.angles(0,1,2) # a simplified version of above
            >>> cell.angles(0,1,3,2) # two angles along path: 0-1-3 and 1-3-2
            >>> cell.angles(tuple(0,1,3,2)) # same as above
            >>> cell.angles((0,1,3),(1,3,2)) # same as above
        """

        v = self.cartesian()
        ids = numpy.asanyarray(ids, dtype=numpy.int64)

        if len(ids.shape) == 1:
            if ids.shape[0] < 3:
                raise ArgumentError("Only %i points are found, at least 3 required" % ids.shape[0])
            vectors = v[ids[:-1], :] - v[ids[1:], :]
            nonzero = numpy.argwhere((vectors ** 2).sum(axis=1) > 0)[:, 0]
            if nonzero.shape[0] == 0:
                raise ArgumentError("All points coincide")

            vectors[:nonzero[0]] = vectors[nonzero[0]]
            vectors[nonzero[-1] + 1:] = vectors[nonzero[-1]]

            vectors_1 = vectors[:-1]
            vectors_2 = -vectors[1:]

            for i in range(nonzero.shape[0] - 1):
                vectors_1[nonzero[i] + 1:nonzero[i + 1]] = vectors_1[nonzero[i]]
                vectors_2[nonzero[i]:nonzero[i + 1] - 1] = vectors_2[nonzero[i + 1] - 1]

        elif len(ids.shape) == 2:
            if ids.shape[1] != 3:
                raise ArgumentError("The input array is [%ix%i], required [nx3]" % ids.shape)
            vectors_1 = v[ids[:, 0], :] - v[ids[:, 1], :]
            vectors_2 = v[ids[:, 2], :] - v[ids[:, 1], :]
        else:
            raise ArgumentError("The input array has unsupported dimensionality %i" % len(ids.shape))

        return __angle__(vectors_1, vectors_2)

    @input_as_list
    def distances(self, ids, threshold=None):
        """
        Computes distances between species and specified points.

        Args:

            ids (array): a list of specimen IDs to compute distances
            between. Several shapes are accepted:

            * empty: returns a 2D matrix of all possible distances
            * nx2 array of ints: returns n distances between each pair
              of [n,0]-[n,1] species;
            * 1D array of ints of length n: returns n-1 distances
              between each pair of [n-1]-[n] species;

            threshold (float): if specified, returns a sparse distance
            matrix with entries less that the threshold. Only for empty
            `ids`;

        Returns:

            A numpy array containing list of distances.
        """

        v = self.cartesian()

        if len(ids) == 0:
            if threshold is not None:
                tree = cKDTree(v)
                return tree.sparse_distance_matrix(tree, max_distance=threshold)
            else:
                return ((v[numpy.newaxis, ...] - v[:, numpy.newaxis, :]) ** 2).sum(axis=-1) ** .5

        ids = numpy.asanyarray(ids, dtype=numpy.int64)

        if len(ids.shape) == 1:
            if ids.shape[0] < 2:
                raise ArgumentError("Only %i points are found, at least 2 required" % ids.shape[0])
            return ((v[ids[:-1], :] - v[ids[1:], :]) ** 2).sum(axis=1) ** .5

        elif len(ids.shape) == 2:
            if ids.shape[1] != 2:
                raise ArgumentError("The input array is [%ix%i], required [nx2]" % ids.shape)
            return ((v[ids[:, 0], :] - v[ids[:, 1], :]) ** 2).sum(axis=1) ** .5

        else:
            raise ArgumentError("The input array has unsupported dimensionality %i" % len(ids.shape))

    def size(self):
        """
        Retrieves the number of points or species in this unit cell.
        
        Returns:
        
            Number of points or species in cell.
        """
        return self.coordinates.shape[0]

    def cartesian(self):
        """
        Computes cartesian coordinates.
        
        Returns:
        
            A numpy array with cartesian coordinates
        """
        return self.transform_to_cartesian(self.coordinates)

    def normalized(self, sort=None):
        """
        Moves all species respecting periodicity so that each
        coordinate becomes in the unit range 0<=x<1 in the cell basis.
        Sorts the data if ``sort`` provided.
        
        Kwargs:
        
            sort: coordinates to sort with: either 'x', 'y', 'z' or 0,1,2
            or a vector in crystal coordiantes to project onto before sorting.
        
        Returns:
        
            A new grid with normalized data.
        """
        sort = __xyz2i__(sort)
        if isinstance(sort, int):
            i = sort
            sort = [0] * self.coordinates.shape[1]
            sort[i] = 1

        result = self.copy()
        result.coordinates = result.coordinates % 1
        if not sort is None:
            result.apply(numpy.argsort(result.coordinates.dot(sort)))

        return result

    def packed(self):
        """
        Moves all species as close to the origin as it is possible. Does
        not perform translation.
        
        Returns:
        
            A new unit cell with packed coordinates.
        """
        result = self.normalized()
        coordinates = result.cartesian()
        vertices = result.vertices()

        d = coordinates[:, numpy.newaxis, :] - vertices[numpy.newaxis, :, :]
        d = (d ** 2).sum(axis=-1)
        d = numpy.argmin(d, axis=-1)

        coordinates -= vertices[d, :]

        result.coordinates = result.transform_from(
            Basis(
                numpy.eye(result.vectors.shape[0])
            ),
            coordinates)

        return result

    @input_as_list
    def isolated(self, gaps, units="crystal"):
        """
        Generates an isolated representation of this cell.
        
        Symmetrically adds vacuum along all unit cell vectors such that
        resulting unit cell vectors are parallel to the initial ones.
        
        Args:
        
            gaps (array): size of the vacuum layer in each direction
            either in cartesian or in crystal units.
            
        Kwargs:
        
            units (str): units of the vacuum size, 'cartesian' or
            'crystal'
            
        Returns:
        
            A new unit cell with spacially isolated species.
        """
        gaps = numpy.asanyarray(gaps, dtype=numpy.float64)
        if units == "cartesian":
            gaps /= ((self.vectors ** 2).sum(axis=1) ** .5)
        elif units == "crystal":
            pass
        else:
            raise ArgumentError("Unknown units: '{}'".format(str(units)))

        result = self.copy()

        gaps += 1
        result.vectors *= gaps[..., numpy.newaxis]
        result.coordinates /= gaps[numpy.newaxis, ...]
        result.coordinates += (0.5 * (gaps - 1) / gaps)[numpy.newaxis, ...]

        return result

    def isolated2(self, gap):
        """
        Generates an isolated representation of this cell.
        
        The resulting cell is rectangular and contains space gaps of at
        least "gap" size.
        
        Args:
        
            gap (float): size of the space gap along all ``self.vectors``.
            
        Returns:
        
            A new unit cell with spacially isolated species.
        """
        c = self.normalized()
        coordinates = c.cartesian() + gap
        shape = numpy.amax(c.vertices(), axis=0) + 2 * gap
        return UnitCell(
            Basis(
                shape,
                kind='orthorombic',
                meta=self.meta,
            ),
            coordinates,
            self.values,
            c_basis='cartesian')

    @input_as_list
    def select(self, piece):
        """
        Selects a piece of this cell.
        
        Args:
        
            piece (array): fraction of the cell to be selected, see
            examples. The order of coordinates in ``piece`` is ``x_from, y_from, ..., z_from, x_to, y_to, ..., z_to``.
            
        Returns:
        
            A numpy array with bools defining whether particular specimen
            is selected or not.
            
        Example:
            
            >>> cell.select((0,0,0,1,1,1)) # select all species with coordinates within (0,1) range
            >>> cell.select(0,0,0,1,1,1) # a simplified version of above
            >>> cell.select(0,0,0,0.5,1,1) # select the 'left' part
            >>> cell.select(0.5,0,0,1,1,1) # select the 'right' part
        """
        if not len(piece) == 2 * self.vectors.shape[0]:
            raise ArgumentError("Wrong coordinates array: expecting {:d} elements, found {:d}".format(
                2 * self.vectors.shape[0], len(piece)
            ))

        piece = numpy.reshape(piece, (2, -1))
        p1 = numpy.amin(piece, axis=0)
        p2 = numpy.amax(piece, axis=0)
        return numpy.all(self.coordinates < p2[numpy.newaxis, :], axis=1) & \
               numpy.all(self.coordinates >= p1[numpy.newaxis, :], axis=1)

    @input_as_list
    def apply(self, selection):
        """
        Applies selection to this cell.
        
        Inverse of ``UnitCell.discard``.
        
        Args:
        
            selection (array): seleted species.
            
        Example:
        
            >>> selection = cell.select((0,0,0,0.5,1,1)) # Selects species in the 'left' part of the unit cell.
            >>> cell.apply(selection) # Applies selection. Species outside the 'left' part are discarded.
        """
        selection = numpy.asanyarray(selection)
        self.coordinates = self.coordinates[selection, :]
        self.values = self.values[selection]

    @input_as_list
    def discard(self, selection):
        """
        Removes specified species from cell.
        
        Inverse of ``Cell.apply``.
        
        Args:
        
            selection (array): species to remove.
            
        Example:
        
            >>> selection = cell.select((0,0,0,0.5,1,1)) # Selects species in the 'left' part of the unit cell.
            >>> cell.discard(selection) # Discards selection. Species inside the 'left' part are removed.
        """
        self.apply(~numpy.asanyarray(selection))

    @input_as_list
    def cut(self, piece, select='auto'):
        """
        Selects a piece of this unit cell and returns it as a smaller
        unit cell.
        
        Args:
        
            piece (array): fraction of the cell to be selected. The order
            of coordinates in ``piece`` is ``x_from, y_from, ..., z_from, x_to, y_to, ..., z_to``.
            
        Kwargs:
        
            select (array): manual selection of points insisde the
            unit cell. By default, selects only those pieces which drop
            into the box defined by `piece`.
            
        Returns:
        
            A smaller unit cell selected.
        """
        if isinstance(select, (str, unicode)) and select == 'auto':
            select = self.select(piece)
        result = self.copy()
        result.apply(select)

        piece = numpy.reshape(piece, (2, -1))
        p1 = numpy.amin(piece, axis=0)
        p2 = numpy.amax(piece, axis=0)

        cartesian_shift = p1.dot(result.vectors)
        cartesian_coords = result.cartesian() - cartesian_shift[numpy.newaxis, :]
        result.vectors *= (p2 - p1)[:, numpy.newaxis]
        result.coordinates = result.transform_from_cartesian(cartesian_coords)
        return result

    @input_as_list
    def add(self, cells):
        """
        Adds species from another unit cells to this one.
        
        Args:
        
            cells (arguments): unit cells to be merged with.
            
        Returns:
        
            A new unit cell with merged data.
        """
        c = [self.coordinates]
        v = [self.values]

        for cell in cells:
            if not numpy.all(cell.vectors == self.vectors):
                raise ArgumentError('Dimension mismatch: %r, %r' % (self.vectors, cell.vectors))
            c.append(cell.coordinates)
            v.append(cell.values)

        return UnitCell(
            self,
            numpy.concatenate(c, axis=0),
            numpy.concatenate(v, axis=0))

    @input_as_list
    def stack(self, cells, vector='x', **kwargs):
        """
        Stacks several cells along one of the vectors.
        
        Args:
        
            cells (list): cells to stack. Corresponding vectors of
            all cells being stacked should match.
            
        Kwargs:
        
            vector (str,int): a vector along which to stack, either 'x',
            'y', 'z' or an int specifying the vector.
            
        The rest of kwargs are redirected to ``Basis.stack``.
        
        Raises:
        
            ArgumentError: in the case of vector mismatch.
            
        Returns:
        
            A bigger unit cell containing all argument cell stacked.
        """
        cells = [self] + cells
        d = __xyz2i__(vector)
        not_d = list(range(self.vectors.shape[0]))
        del not_d[d]
        dims = self.vectors.shape[0]

        for c in cells:
            if not isinstance(c, Basis):
                raise ArgumentError('The object {} is not an instance of a Basis'.format(c))

        basis = Basis.stack(*cells, vector=vector, **kwargs)

        values = numpy.concatenate(tuple(cell.values for cell in cells if isinstance(cell, UnitCell)), axis=0)

        coordinates = []
        shift = numpy.zeros(dims)
        for c in cells:
            if isinstance(c, UnitCell):
                # Fix for not-excatly-the-same vectors
                original = c.vectors[not_d].copy()
                c.vectors[not_d] = self.vectors[not_d]
                coordinates.append(c.cartesian() + shift[numpy.newaxis, :])
                c.vectors[not_d] = original
            shift += c.vectors[d, :]
        coordinates = numpy.concatenate(coordinates, axis=0)

        return UnitCell(basis, coordinates, values, c_basis="cartesian")

    @input_as_list
    def supercell(self, vec):
        """
        Produces a supercell from a given unit cell.
        
        Args:
        
            vec (array): the supercell vectors in units of current unit
            cell vectors
            
        Returns:
        
            A new supercell.
        """
        vec = numpy.asanyarray(vec, dtype=numpy.float64)

        sc_min = None
        sc_max = None

        for v in itertools.product((0.0, 1.0), repeat=vec.shape[0]):

            vertex = (vec * numpy.asanyarray(v)[:, numpy.newaxis]).sum(axis=0)

            if sc_min is None:
                sc_min = numpy.asanyarray(v)
            else:
                sc_min = numpy.minimum(sc_min, vertex)

            if sc_max is None:
                sc_max = numpy.asanyarray(v)
            else:
                sc_max = numpy.maximum(sc_max, vertex)

        # Fix roundoff
        random_displacement = random.rand(self.coordinates.shape[-1])[numpy.newaxis, :]
        u = self.copy()
        u.coordinates += random_displacement

        sc = u.repeated((sc_max - sc_min).astype(numpy.int64)).normalized()

        origin = (self.vectors * sc_min[:, numpy.newaxis]).sum(axis=0)

        result = UnitCell(
            Basis(
                numpy.dot(vec, self.vectors),
                meta=self.meta),
            sc.cartesian() + origin[numpy.newaxis, :],
            sc.values,
            c_basis='cartesian',
        )

        result.apply(numpy.all(numpy.logical_and(result.coordinates >= 0, result.coordinates < 1), axis=1))
        result.coordinates -= result.transform_from(u, random_displacement)
        return result.normalized()

    def species(self):
        """
        Collects number of species of each kind in this cell.
        
        Particularly useful for counting the number of atoms.
        
        Returns:
        
            A dictionary containing species as keys and number of atoms
            as values.
        """
        answer = {}
        for s in self.values:
            try:
                answer[s] += 1
            except:
                answer[s] = 1
        return answer

    @input_as_list
    def reorder_vectors(self, new):
        """
        Reorders vectors.
        
        Args:
        
            new (array): new mapping of vectors.
            
        Example:
        
            >>> cell.reorder_vectors(0, 1, 2) # does nothing
            >>> cell.reorder_vectors(1, 0, 2) # swaps first and second vectors.
            
        .. note::
        
            A call to this method does not modify the output of ``self.cartesian()``.
        """
        new = tuple(__xyz2i__(i) for i in new)
        Basis.reorder_vectors(self, new)
        self.coordinates = self.coordinates[:, new]

    def as_grid(self, fill=float("nan")):
        """
        Converts this unit cell to grid.
        
        Kwargs:
        
            fill: default value to fill with;
        
        Returns:
        
            A new grid with data from initial cell.
        """

        # Convert coordinates
        coordinates = list(
            numpy.sort(
                numpy.unique(self.coordinates[:, i])
            ) for i in range(self.coordinates.shape[1])
        )

        # A coordinates lookup table
        coord2index = list(
            dict(zip(a, range(a.size))) for a in coordinates
        )

        # Convert values
        data = fill * numpy.ones(tuple(a.size for a in coordinates) + self.values.shape[1:], dtype=self.values.dtype)

        for c, v in zip(self.coordinates, self.values):
            indexes = tuple(coord2index[i][cc] for i, cc in enumerate(c))
            data[indexes] = v

        return Grid(
            self,
            coordinates,
            data,
        )

    @input_as_list
    def interpolate(self, points, driver=None, periodic=True, **kwargs):
        """
        Interpolates values at specified points. By default uses
        ``scipy.interpolate.griddata``.
        
        Args:
        
            points (array): points to interpolate at.
            
        Kwargs:
        
            driver (func): interpolation driver.
            
            periodic (bool): employs periodicity of a unit cell.
            
            kwargs: keywords to the driver.
            
        Returns:
        
            A new unit cell with interpolated data.
            
        """
        points = numpy.asanyarray(points, dtype=numpy.float64)

        if driver is None:
            from scipy import interpolate
            driver = interpolate.griddata

        if periodic:

            # Avoid edge problems by creating copies of this cell
            supercell = self.repeated((3,) * self.vectors.shape[0]).normalized()

            data_points = supercell.cartesian()
            data_values = supercell.values

            # Shift points to the central unit cell
            points_i = self.transform_to_cartesian(points % 1) + self.vectors.sum(axis=0)[numpy.newaxis, :]

        else:

            data_points = self.cartesian()
            data_values = self.values
            points_i = self.transform_to_cartesian(points)

        # Interpolate
        return UnitCell(
            self,
            points,
            cast_units(driver(data_points, data_values, points_i, **kwargs), self.values),
        )


class Grid(Basis):
    """
    A class describing a data on a grid in a periodic environment.
    
    Args:
    
        vectors (Basis,array): a crystal basis.
        
        coordinates (array): a list of arrays of coordinates specifying
        grid.
        
        values (array): a multidimensional array with data on the grid.
        
    Kwargs:

        meta (dict): a metadata for this Grid;
    """

    def __init__(self, vectors, coordinates, values, meta=None):

        if isinstance(vectors, Basis):
            Basis.__init__(self, vectors.vectors, meta=vectors.meta)
        else:
            Basis.__init__(self, vectors)
        if meta is not None:
            self.meta.update(meta)

        dims = self.vectors.shape[0]
        self.coordinates = list(numpy.asanyarray(c, dtype=numpy.float64) for c in coordinates)
        self.values = numpy.asanyarray(values)

        # Proceed to checks
        if not len(self.coordinates) == dims:
            raise ArgumentError(
                "The size of the basis is {:d} but the number of coordinates specified is different: {:d}".format(
                    dims, len(self.coordinates)
                )
            )

        for i, c in enumerate(self.coordinates):
            if not len(c.shape) == 1:
                raise ArgumentError("Coordinates[{:d}] is not a 1D array".format(i))

        if len(self.values.shape) < dims:
            raise ArgumentError("The dimensionality of a 'values' array is less ({:d}) than expected ({:d})".format(
                len(self.values.shape), dims))

        for i in range(dims):
            if not self.values.shape[i] == self.coordinates[i].shape[0]:
                raise ArgumentError(
                    "The {:d} dimension of 'values' array is equal to {:d} which is different from the size of a corresponding 'coordinates' array {:d}".format(
                        i, self.values.shape[i], self.coordinates[i].shape[0]))

    def __getstate__(self):
        result = super(Grid, self).__getstate__()
        result.update(dict(
            coordinates=tuple(i.copy() for i in self.coordinates),
            values=self.values.copy(),
        ))
        return result

    def __setstate__(self, data):
        super(Grid, self).__setstate__(data)
        self.__init__(self, data["coordinates"], data["values"])

    def __eq__(self, another):
        result = Basis.__eq__(self, another)
        result = result and numpy.all(self.values == another.values)
        for i, j in zip(self.coordinates, another.coordinates):
            result = result and numpy.all(i == j)
        return result

    def size(self):
        """
        Retrieves the total size of points on the grid.
        
        Returns:
        
            Number of species in cell as an integer.
        """
        r = 1
        for a in self.coordinates:
            r *= a.size
        return r

    @staticmethod
    def combine_arrays(arrays):
        """
        Transforms input 1D arrays of coordinates into (N+1)D mesh array
        where first N dimensions correspond to a particular grid point
        and the last dimension specifies all coordinates of this grid point.
        
        Args:
        
            arrays (list): a list of 1D arrays;
            
        Returns:
        
            A meshgrid array with coordinates.
        """
        mg = numpy.meshgrid(*arrays, indexing='ij')
        return numpy.concatenate(tuple(i[..., numpy.newaxis] for i in mg), axis=len(mg))

    @staticmethod
    def uniform(size, endpoint=False):
        """
        Transform positive integers `size` into a meshgrid array
        representing a grid where grid points span uniformly zero to one
        intervals.
        
        Args:
        
            size (array): an array with positive integers;
            
        Kwargs:
        
            endpoint (bool): indicates whether to include x=1 into grids.
            
        Returns:
        
            A meshgrid array with coordinates.
        """
        return Grid.combine_arrays(tuple(
            numpy.linspace(0, 1, i, endpoint=endpoint) for i in size
        ))

    def explicit_coordinates(self):
        """
        Creates an (N+1)D array with explicit coordinates at each grid
        point.
        
        Returns:
        
            An (N+1)D array with coordinates.
        """
        return Grid.combine_arrays(self.coordinates)

    def cartesian(self):
        """
        Computes cartesian coordinates.
        
        Returns:
        
            A multidimensional numpy array with cartesian coordinates at
            each grid point.
        """
        return self.transform_to_cartesian(self.explicit_coordinates())

    def normalized(self):
        """
        Moves all grid points respecting periodicity so that each
        coordinate becomes in the unit range 0<=x<1 in the cell basis.
        Sorts the data.
        
        Returns:
        
            A new grid with normalized data.
        """
        result = self.copy()

        for i in range(len(result.coordinates)):
            result.coordinates[i] = result.coordinates[i] % 1

        result.apply(tuple(numpy.argsort(a) for a in result.coordinates))

        return result

    @input_as_list
    def isolated(self, gaps, units="cartesian"):
        """
        Generates an isolated representation of this grid.
        
        Symmetrically adds vacuum along all basis vectors such that
        resulting grid basis vectors are parallel to the initial ones.
        
        Args:
        
            gaps (array): size of the vacuum layer in each direction
            either in cartesian or in crystal units.
            
        Kwargs:
        
            units (str): units of the vacuum size, 'cartesian' or
            'crystal'
            
        Returns:
        
            A new isolated grid.
        """
        gaps = numpy.asanyarray(gaps, dtype=numpy.float64)
        if units == "cartesian":
            gaps /= ((self.vectors ** 2).sum(axis=1) ** .5)
        elif units == "crystal":
            pass
        else:
            raise ArgumentError("Unknown units: '{}'".format(str(units)))

        result = self.copy()

        gaps += 1
        result.vectors *= gaps[..., numpy.newaxis]

        for i in range(len(result.coordinates)):
            result.coordinates[i] /= gaps[i]
            result.coordinates[i] += (0.5 * (gaps[i] - 1) / gaps[i])

        return result

    @input_as_list
    def select(self, piece):
        """
        Selects a piece of this grid.
        
        Args:
        
            piece (array): fraction of the grid to be selected, see
            examples. The order of coordinates in ``piece`` is ``x_from, y_from, ..., z_from, x_to, y_to, ..., z_to``.
            
        Returns:
        
            A list of numpy arrays with bools defining whether particular
            grid point is selected or not.
            
        Example:
            
            >>> grid.select((0,0,0,1,1,1)) # select all grid points with coordinates within (0,1) range
            >>> grid.select(0,0,0,1,1,1) # a simplified version of above
            >>> grid.select(0,0,0,0.5,1,1) # select the 'left' part
            >>> grid.select(0.5,0,0,1,1,1) # select the 'right' part
        """
        if not len(piece) == 2 * self.vectors.shape[0]:
            raise ArgumentError("Wrong coordinates array: expecting {:d} elements, found {:d}".format(
                2 * self.vectors.shape[0], len(piece)
            ))

        piece = numpy.reshape(piece, (2, -1))
        p1 = numpy.amin(piece, axis=0)
        p2 = numpy.amax(piece, axis=0)
        return list((c < mx) & (c >= mn) for c, mn, mx in zip(self.coordinates, p1, p2))

    @input_as_list
    def apply(self, selection):
        """
        Applies selection to this grid.
        
        Inverse of ``Grid.discard``.
        
        Args:
        
            selection (array): seleted grid points.
            
        Example:
        
            >>> selection = grid.select((0,0,0,0.5,1,1)) # Selects species in the 'left' part of the grid.
            >>> grid.apply(selection) # Applies selection. Species outside the 'left' part are discarded.
        """
        selection = list(selection)
        slices = [slice(None, None, None)] * len(self.coordinates)

        for i in range(len(self.coordinates)):

            if not isinstance(selection[i], slice):
                selection[i] = numpy.asanyarray(selection[i])
            self.coordinates[i] = self.coordinates[i][selection[i]]

            # Set a valid slice
            slices[i] = selection[i]
            # Apply slice
            self.values = self.values[tuple(slices)]
            # Revert slice
            slices[i] = slice(None, None, None)

    @input_as_list
    def discard(self, selection):
        """
        Removes specified points from this grid.
        
        Inverse of ``Grid.apply``.
        
        Args:
        
            selection (array): points to remove.
            
        Example:
        
            >>> selection = grid.select((0,0,0,0.5,1,1)) # Selects points in the 'left' part of the grid.
            >>> grid.discard(selection) # Discards selection. Points inside the 'left' part are removed.
        """
        self.apply(tuple(~numpy.asanyarray(i) for i in selection))

    @input_as_list
    def cut(self, piece):
        """
        Selects a piece of the grid and returns it as a smaller basis.
        
        Kwargs:
        
            piece (array): fraction of the grid to be selected. The order
            of coordinates in ``piece`` is ``x_from, y_from, ..., z_from, x_to, y_to, ..., z_to``.
            
        Returns:
        
            A smaller grid selected.
        """
        result = self.copy()
        result.apply(result.select(piece))

        piece = numpy.reshape(piece, (2, -1))
        p1 = numpy.amin(piece, axis=0)
        p2 = numpy.amax(piece, axis=0)

        for i in range(len(result.coordinates)):
            result.coordinates[i] -= p1[i]
            result.coordinates[i] /= (p2 - p1)[i]
        result.vectors *= (p2 - p1)[numpy.newaxis, :]
        return result

    @input_as_list
    def add(self, grids, fill=float("nan")):
        """
        Adds grid points from another grids to this one.
        
        Args:
        
            grids (arguments): grids to be merged with.
            
        Returns:
        
            A new grid with merged data.
        """
        dims = len(self.coordinates)
        grids = [self] + grids
        new_coordinates = []

        # Coordinates lookup tables
        coord2index = []

        # Calculate unique coordinates on the grid and lookup tables
        for j in range(dims):

            c = []
            for i in grids:
                c.append(i.coordinates[j])

            c = numpy.concatenate(c, axis=0)
            unique_coordinates, lookup = numpy.unique(c, return_inverse=True)
            new_coordinates.append(unique_coordinates)
            coord2index.append(lookup)

        new_shape = tuple(a.shape[0] for a in new_coordinates)
        new_values = numpy.ones(new_shape + self.values.shape[dims:]) * fill

        # Fill in the values
        offsets = [0] * dims
        for i in grids:

            location = tuple(c2i[o:o + c.shape[0]] for o, c2i, c in zip(offsets, coord2index, i.coordinates))
            location = numpy.ix_(*location)
            new_values[location] = i.values

            for j in range(len(offsets)):
                offsets[j] += i.coordinates[j].shape[0]

        return Grid(
            self,
            new_coordinates,
            new_values,
        )

    @input_as_list
    def stack(self, grids, vector='x', **kwargs):
        """
        Stacks several grids along one of the vectors.
        
        Args:
        
            grids (list): grids to stack. Corresponding vectors of
            all grids being stacked should match.
            
        Kwargs:
        
            vector (str,int): a vector along which to stack, either 'x',
            'y', 'z' or an int specifying the vector.
        
        The rest of kwargs are redirected to ``Basis.stack``.
                    
        Raises:
        
            ArgumentError: in the case of vector mismatch.
            
        Returns:
        
            A bigger grid containing all argument grids stacked.
        """
        grids = [self] + grids
        d = __xyz2i__(vector)
        dims = self.vectors.shape[0]

        otherVectors = list(range(grids[0].vectors.shape[0]))
        del otherVectors[d]

        basis = Basis.stack(*grids, vector=vector, **kwargs)

        for g in grids:
            if not isinstance(g, Basis):
                raise ArgumentError('The object {} is not an instance of a Basis'.format(g))

        for i, g in enumerate(grids[1:]):
            if isinstance(g, Grid):
                for dim in otherVectors:
                    if not len(g.coordinates[dim]) == len(self.coordinates[dim]) or not numpy.all(
                            g.coordinates[dim] == self.coordinates[dim]):
                        raise ArgumentError(
                            "Grid coordinates in dimension {:d} of cells 0 and {:d} are different".format(dim, i))

        values = numpy.concatenate(tuple(grid.values for grid in grids if isinstance(grid, Grid)), axis=d)

        stackingVectorsLen = numpy.asanyarray(tuple((grid.vectors[d] ** 2).sum(axis=-1) ** .5 for grid in grids))
        shifts = numpy.cumsum(stackingVectorsLen)
        shifts = shifts / shifts[-1]

        # kx+b
        k = numpy.ones((len(grids), dims))
        k[:, d] = stackingVectorsLen / stackingVectorsLen.sum()
        b = numpy.zeros((len(grids), dims))
        b[:, d] = numpy.concatenate(((0,), shifts[:-1]))

        coordinates = []
        for dim in range(dims):
            if dim == d:
                coordinates.append(numpy.concatenate(tuple(
                    grid.coordinates[dim] * k[i, dim] + b[i, dim] for i, grid in enumerate(grids) if
                    isinstance(grid, Grid)
                ), axis=0))
            else:
                coordinates.append(self.coordinates[dim])

        return Grid(basis, coordinates, values)

    @input_as_list
    def reorder_vectors(self, new):
        """
        Reorders vectors. Does not change output of ``Grid.cartesian()``.
        
        Args:
        
            new (array): new mapping of vectors.
            
        Example:
        
            >>> grid.reorder_vectors(0, 1, 2) # does nothing
            >>> grid.reorder_vectors(1, 0, 2) # swaps first and second vectors.
            
        .. note::
        
            A call to this method does not modify the output of ``self.cartesian()``.
        """
        new = list(__xyz2i__(i) for i in new)
        Basis.reorder_vectors(self, new)
        self.coordinates = list(self.coordinates[n] for n in new)

        # Change values using swapaxes
        for i in range(len(new)):
            if not new[i] == i:
                self.values = self.values.swapaxes(i, new[i])
                new[new[i]] = new[i]

    def as_unitCell(self):
        """
        Converts this cell into a ``UnitCell``.
        
        Returns:
        
            A new ``UnitCell``.
        """
        c = self.explicit_coordinates()
        c = c.reshape((-1, c.shape[-1]))
        v = self.values.reshape((-1,) + self.values.shape[len(self.coordinates):])

        return UnitCell(self, c, v)

    def interpolate_to_array(self, points, driver=None, periodic=True, **kwargs):
        """
        Interpolates values at specified points and returns an array of
        interpolated values. By default uses ``scipy.interpolate.interpn``.
        
        Args:
        
            points (array): points to interpolate at.
            
        Kwargs:
        
            driver (func): interpolation driver;
            
            periodic (bool): employs periodicity of a unit cell;
            
            The rest of keyword arguments are passed to the driver.
            
        Returns:
        
            An array with values of corresponding shape.
        """

        if driver is None:
            from scipy import interpolate
            driver = interpolate.interpn

        points = numpy.asanyarray(points)
        normalized = self.normalized()

        if periodic:

            data_points = normalized.coordinates
            data_values = normalized.values

            # Avoid edge problems
            for i, a in enumerate(data_points):
                data_points[i] = numpy.insert(a, (0, a.size), (a[-1] - 1.0, a[0] + 1.0))

                left_slice = (slice(None),) * i + ((0,),) + (slice(None),) * (len(data_points) - i - 1)
                left = data_values[left_slice]

                right_slice = (slice(None),) * i + ((-1,),) + (slice(None),) * (len(data_points) - i - 1)
                right = data_values[right_slice]

                data_values = numpy.concatenate((right, data_values, left), axis=i)

            points = points % 1

        else:

            data_points = normalized.coordinates
            data_values = normalized.values

        # Interpolate
        return cast_units(driver(data_points, data_values, points, **kwargs), self.values)

    def interpolate_to_grid(self, points, **kwargs):
        """
        Interpolates values at specified points and returns a grid.
        By default uses ``scipy.interpolate.interpn``.
        
        Args:
        
            points (array): points to interpolate at.
            
        Kwargs are passed to ``self.interpolate_to_array``.
            
        Returns:
        
            A grid with interpolated values.
        """
        return Grid(self, points, self.interpolate_to_array(Grid.combine_arrays(points), **kwargs))

    def interpolate_to_cell(self, points, **kwargs):
        """
        Interpolates values at specified points and returns a unit cell.
        By default uses ``scipy.interpolate.interpn``.
        
        Args:
        
            points (array): points to interpolate at.
            
        Kwargs are passed to ``self.interpolate_to_array``.
            
        Returns:
        
            A unit cell interpolated values.
        """
        return UnitCell(self, points, self.interpolate_to_array(points, **kwargs))

    def interpolate_to_path(self, points, n, anchor=True, **kwargs):
        """
        Interpolates values to a path: performs path generation using
        ``Basis.generate_path`` and interpolates values on it.
        
        Args:
        
            points (array): key points of the path expressed in lattice coordinates;
            
            n (int): the total number of points on the path.
            
        Kwargs:
        
            anchor (bool): force the specified points to be present in
            the final path. If True alters slightly the total point number;
            
            kwargs: keywords to the 'interpolate_to_cell' routine.
            
        Returns:
        
            A unit cell interpolated values.
        """
        return self.interpolate_to_cell(self.generate_path(points, n, anchor=anchor), **kwargs)

    def tetrahedron_density(self, points, resolved=False, weights=None):
        """
        Convolves data to calculate density (of states). Uses the
        tetrahedron method from PRB 49, 16223 by E. Blochl et al. Works
        only in a 3D space.
        
        Args:
        
            points (array): values to calculate density at.
        
        Kwargs:
        
            resolved (bool): if True returns a spacially and index
            resolved density. The dimensions of the returned array
            are ``self.values.shape + points.shape``.
            
            weights (array): if specified and ``resolved`` is False
            convolves result with the specified weights.
            
        Returns:
        
            A numpy array containing density: 1D if ``resolved == False``
            or a corresponding Grid if ``resolved == True``.
        """
        if not self.vectors.shape[0] == 3:
            raise ArgumentError("The tetrahedron density method is implemented only for 3D grids")

        initial = self.values
        points = numpy.asanyarray(points, dtype=numpy.float64)
        self.values = numpy.reshape(self.values, self.values.shape[:3] + (-1,))
        minima = numpy.min(self.values, axis=(0, 1, 2))
        maxima = numpy.max(self.values, axis=(0, 1, 2))
        bottom = (maxima < points.min()).sum()
        top = (minima < points.max()).sum()

        # Optimize size
        self.values = self.values[..., bottom:top]

        if resolved:

            # Sum over bands
            raw = tetrahedron(self, points).reshape(self.values.shape + points.shape)
            self.values = initial
            return Grid(
                self,
                self.coordinates,
                raw,
            )

        else:

            if weights is None:
                weights = numpy.ones(self.values.shape, dtype=numpy.float64)

            else:
                weights = numpy.asanyarray(weights, dtype=numpy.float64)
                weights = numpy.reshape(weights, weights.shape[:3] + (-1,))[..., bottom:top]

            raw = tetrahedron_plain(self, points, weights)
            self.values = initial
            return raw
