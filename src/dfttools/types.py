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

# This is here to determine the default string data type in numpy.
element_type = numpy.array("Ca").dtype


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
        v1, v2 (ndarray): arrays of the same size with vectors' coordinates;
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
        vectors (Iterable, ndarray, Basis): another Basis or a 2D matrix of basis
        vectors;
        kind (str): shortcut keyword for several most common basis sets
            * 'default': expects `vectors` to be a 2D array with basis
              vectors in cartesian coordinates;
            * 'orthorhombic': expects `vectors` to be a 1D array with
              dimensions of an orthorhombic basis set;
            * 'triclinic': expects `vectors` to be a 1D array with 3
              lengths of edges and 3 cosines of face angles;
        meta (dict): metadata for this Basis.
    """
    def __init__(self, vectors, kind='default', meta=None):

        if isinstance(vectors, Basis):
            self.vectors = numpy.asanyarray(vectors.vectors)
            _meta = vectors.meta.copy()
            if meta is not None:
                _meta.update(meta)

        else:
            vectors = numpy.asanyarray(vectors, dtype=numpy.float64)
            _meta = meta.copy() if meta is not None else {}

            if kind == 'default':
                self.vectors = vectors

            elif kind == 'orthorhombic':
                self.vectors = cast_units(numpy.diag(vectors), vectors)

            elif kind == 'triclinic':
                lengths = vectors[0:3]
                cosines = vectors[3:]
                volume = lengths[0] * lengths[1] * lengths[2] * (
                    1 + 2 * cosines[0] * cosines[1] * cosines[2] - cosines[0] ** 2 - cosines[1] ** 2 - cosines[2] ** 2
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

        self.meta = _meta

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
        Restores the Basis from JSON data.

        Args:
            j (dict): JSON data;

        Returns:
            A Basis object.
        """
        j = dict(j)
        if "type" not in j or j["type"] != cls.class_id():
            raise TypeError("Invalid JSON, expected type {}".format(cls.class_id()))
        del j["type"]
        result = cls(**j)
        return result

    def transform_to(self, basis, coordinates):
        """
        Transforms coordinates to another basis set.

        Args:
            basis (Basis): new basis to transform to;
            coordinates (Iterable, ndarray): array of coordinates to be
            transformed;

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
            basis (Basis): basis to transform from;
            coordinates (Iterable, ndarray): array of coordinates to be
            transformed;

        Returns:

            An array with transformed coordinates.
        """
        return basis.transform_to(self, coordinates)

    def transform_to_cartesian(self, coordinates):
        """
        Transforms coordinates to cartesian.

        Args:
            coordinates (Iterable, ndarray): array of coordinates to be
            transformed;

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
            coordinates (Iterable, ndarray): array of coordinates to be
            transformed;

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
            axis (Iterable, ndarray): axis to rotate around;
            angle (float): angle to rotate;
            units (str): units of the angle: 'rad', 'deg' or 'frac';

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
        rot_matrix = c * numpy.eye(self.vectors.shape[0]) + s * axis_x + (1 - c) * numpy.dot(axis[:, numpy.newaxis],
                                                                                             axis[numpy.newaxis, :])
        result = self.copy()
        result.vectors = numpy.dot(result.vectors, rot_matrix)
        return result

    @property
    def volume(self):
        """
        The volume of the unit cell formed by this basis.
        """
        return abs(numpy.linalg.det(self.vectors))

    def reciprocal(self):
        """
        Computes a reciprocal basis.

        Returns:
            Reciprocal basis.

        .. note::
            The :math:`2 \\pi` prefactor is not included.
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

    def copy(self):
        """
        Calculates a copy.

        Returns:
            A deep copy of self.
        """
        return self.from_json(self.to_json())

    @input_as_list
    def stack(self, other, vector='x', tolerance=1e-10, restrict_collinear=False):
        """
        Stacks several instances along one of the vectors.

        Args:
            other (Iterable): other instances to stack. The corresponding vectors
            of all instances being stacked should match against each other;
            vector (str, int): the vector along which to stack, either 'x',
            'y', 'z' or an int specifying the vector;
            tolerance (float): the error threshold when determining equality of
            other (non-stacking) vectors;
            restrict_collinear (bool): if True will perform a check whether stacking
            vectors are collinear and raise ValueError if they are not;

        Raises:
            ValueError: In case of non-collinear or non-matching vectors;

        Returns:
            A stacked composition of instances (Basis, UnitCell, Grid or other)..
        """
        other = [self] + other
        d = __xyz2i__(vector)

        other_vectors = list(range(other[0].vectors.shape[0]))
        del other_vectors[d]

        # 3d array with lattice vectors: shapes[i,j,k] i=cell, j=lattice vector, k = component
        shapes = numpy.concatenate(tuple(
            i.vectors[numpy.newaxis, ...] for i in other
        ), axis=0)

        # Check if non-stacking lattice vectors coincide
        stacking_vectors_sum = shapes[:, d, :].sum(axis=0)
        vec_lengths = (shapes ** 2).sum(axis=2) ** 0.5
        other_vectors_d = shapes[:, other_vectors, :] - shapes[0, other_vectors, :][numpy.newaxis, ...]
        other_vectors_ds = (other_vectors_d ** 2).sum(axis=-1) ** .5

        if numpy.any(other_vectors_ds > tolerance * vec_lengths[:, other_vectors]):
            raise ArgumentError(
                'Dimension mismatch for stacking:\n{}\nCheck your input basis vectors or set tolerance to at least {} '
                'to silence this exception'.format(
                    shapes,
                    numpy.amax(other_vectors_ds / vec_lengths[:, other_vectors]),
                ))

        if restrict_collinear and numpy.any(
                numpy.abs(__angle__(shapes[:, d, :], stacking_vectors_sum[numpy.newaxis, :]) - 1) > tolerance):
            raise ArgumentError(
                'Vectors to stack along are not collinear:\n{}\nCheck your input basis vectors or set tolerance to at '
                'least {} to silence this exception'.format(
                    shapes,
                    numpy.amax(numpy.abs(__angle__(shapes[:, d, :], stacking_vectors_sum[numpy.newaxis, :]) - 1)),
                ))

        shape = self.vectors.copy()
        shape[d, :] = stacking_vectors_sum

        return Basis(shape, meta=self.meta)

    @input_as_list
    def repeated(self, times):
        """
        Produces a new instance from a given one by repeating it along
        all vectors the given numbers of times.

        Args:
            times (Iterable, ndarray): array of ints specifying
            repetition counts;

        Returns:
            An instance (Basis, UnitCell, Grid or other) repeated along
            all dimensions.
        """
        c = self
        for i, t in enumerate(times):
            c = c.stack(*((c,) * (t - 1)), vector=i)
        return c

    @input_as_list
    def transpose_vectors(self, new):
        """
        Transposes basis vectors inplace.

        Args:
            new (Iterable, ndarray): the new order of vectors;

        Example:
            >>> basis = Basis((1, 2, 3), kind='orthorhombic')
            >>> basis.transpose_vectors(0, 1, 2) # does nothing
            >>> basis.transpose_vectors(1, 0, 2) # swaps first and second vectors.
        """
        new = tuple(__xyz2i__(i) for i in new)

        if not len(new) == self.vectors.shape[0]:
            raise ArgumentError(
                "The new mapping of vectors should be of the size of {:d}".format(self.vectors.shape[0]))

        if not len(set(new)) == self.vectors.shape[0]:
            raise ArgumentError("The input contains duplicates")

        self.vectors = self.vectors[new, :]

    def generate_path(self, points, n, skip_segments=None):
        """
        Generates a piecewise-linear path through points specified.
        Useful for constructing band-structure paths.

        Args:
            points (Iterable, ndarray): path milestones;
            n (int): the total number of points on the path;
            skip_segments (Iterable): optional segments to skip;

        Returns:
            Path coordinates expressed in this basis.

        .. note::
            Milestone points are always present in the path
            returned.
        """

        def interpolate(_p1, _p2, _n, _e):
            x = numpy.linspace(0, 1, _n + 2)[:, numpy.newaxis]
            if not _e:
                x = x[:-1]
            return _p1[numpy.newaxis, :] * (1 - x) + _p2[numpy.newaxis] * x

        if skip_segments is None:
            skip_segments = tuple()
        skip_segments = numpy.array(skip_segments, dtype=int)

        points = numpy.asanyarray(points)
        lengths = numpy.linalg.norm(self.transform_to_cartesian(points[:-1] - points[1:]), axis=1)

        mask_segment = numpy.ones(len(points), dtype=bool)
        mask_segment[skip_segments] = False
        mask_segment[-1] = False
        n_reserved = (numpy.logical_or(mask_segment[1:], mask_segment[:-1]).sum())
        n_reserved += mask_segment[0]

        if n_reserved == 0:
            raise ValueError("Empty edges specified")

        if n < n_reserved:
            raise ValueError("The number of points is less then the number of edges {:d} < {:d}".format(n, n_reserved))

        mask_endpoint = numpy.logical_not(mask_segment[1:])
        mask_segment = mask_segment[:-1]

        points_l = points[:-1][mask_segment]
        points_r = points[1:][mask_segment]
        lengths = lengths[mask_segment]
        buckets = numpy.zeros(len(lengths), dtype=int)
        endpoints = mask_endpoint[mask_segment]
        for i in range(n - n_reserved):
            dl = lengths / (buckets + 1)
            buckets[numpy.argmax(dl)] += 1
        result = []
        for pt1, pt2, _n, e in zip(points_l, points_r, buckets, endpoints):
            result.append(interpolate(pt1, pt2, _n, e))
        return numpy.concatenate(result)


def diamond_basis(a):
    """
    Generates crystal basis of diamond.

    Args:
        a (float): the lattice constant;

    Returns:
        The diamond basis.
    """
    a = 0.5 * a
    return Basis([[0, a, a], [a, 0, a], [a, a, 0]])


class UnitCell(Basis):
    """
    A class describing an arbitrary set of points in a periodic environment.
    Useful for constructing crystal structures, band paths, etc.

    Args:
        vectors (Basis, Iterable, ndarray): basis vectors;
        coordinates (Iterable, ndarray): 2D array of points' coordinates;
        values (Iterable, ndarray): values for each point specified;
        meta (dict): metadata for this UnitCell;
        c_basis (str, Basis): Basis for the input coordinates or 'cartesian'
        if coordinates are passed in the cartesian basis;
        dtype (type): enforcing data type for `values` array;
    """
    def __init__(self, vectors, coordinates, values, meta=None, c_basis=None, dtype=None):

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
                    'Coordinates array is 2D but the last dimension {:d} is not equal to the space '
                    'dimensionality {:d}'.format(self.coordinates.shape[1], dims))

        # Coordinates are now prepared, proceed to values
        self.values = numpy.asanyarray(values, dtype=dtype)
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
        if c_basis is None:
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
        Computes angles between points in this cell.

        Args:
            ids (Iterable, ndarray): point IDs to compute angles between.
            Several shapes are accepted:
                * nx3 array: computes n cosines of angles [i, 0]-[i, 1]-[i, 2];
                * 1D array of length n: computes n-2 cosines of angles along
                  the path ...-[i-1]-[i]-[i+1]-...;

        Returns:
            An array with cosines.

        Example:
            >>> cell = UnitCell(Basis((1, 2, 3), kind="orthorhombic"), numpy.random.rand((4, 3)), numpy.arange(4))
            >>> cell.angles((0, 1, 2)) # angle between vectors connecting {second and first} and {second and third} pts
            >>> cell.angles(0, 1, 2) # a simplified version of the above
            >>> cell.angles(0, 1, 3, 2) # two angles along path: 0-1-3 and 1-3-2
            >>> cell.angles(tuple(0, 1, 3, 2)) # same as the above
            >>> cell.angles((0, 1, 3),(1, 3, 2)) # same as the above
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
        Computes distances between points in this cell.

        Args:
            ids (Iterable, ndarray): specimen IDs to compute distances
            between. Several shapes are accepted:
                * *empty*: returns a 2D matrix of all possible distances
                * nx2 array of ints: returns n distances between each pair
                  of [i, 0]-[i, 1] species;
                * 1D array of ints of length n: returns n-1 distances
                  between each pair of [i-1]-[i] species;
            threshold (float): if specified, returns a sparse distance
            matrix with entries less than the threshold. Only for empty
            `ids`;

        Returns:
            A numpy array with distances.
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

    @property
    def size(self):
        """
        The number of points in this cell.
        """
        return self.coordinates.shape[0]

    def cartesian(self):
        """
        Computes cartesian coordinates of points in this cell.

        Returns:
            A numpy array with cartesian coordinates.
        """
        return self.transform_to_cartesian(self.coordinates)

    def normalized(self, sort=None):
        """
        Moves all points to their periodic images in the 0-1 coordinate range.
        Sorts the data if `sort` provided.

        Args:
            sort (str, int, Iterable, ndarray): the axis to sort along: either
            'x', 'y', 'z', or 0, 1, 2, or an arbitrary vector in crystal
            coordinates to project onto for sorting;

        Returns:
            A new cell with the normalized data.
        """
        sort = __xyz2i__(sort)
        if isinstance(sort, int):
            i = sort
            sort = [0] * self.coordinates.shape[1]
            sort[i] = 1

        result = self.copy()
        result.coordinates = result.coordinates % 1
        if sort is not None:
            result.apply(numpy.argsort(result.coordinates.dot(sort)))

        return result

    def ws_packed(self):
        """
        Generates a new cell where all points are replaced by
        their periodic images closest to the origin (i.e. appear
        inside Wigner-Seitz cell).

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
        Isolates points from their images in this cell or grid by
        elongating basis vectors while keeping distances between the
        points fixed.

        Args:
            gaps (Iterable, ndarray): the elongation amount in cartesian
            or in crystal units;
            units (str): units of `gaps`: 'cartesian' or 'crystal';

        Returns:
            A bigger cell where points are spatially isolated from their
            images.
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
        Isolates points from their images in this cell by constructing a new
        larger orthorhombic cell.
        Args:
            gap (float): the minimal gap size between the cloud of points and
            its periodic images;

        Returns:
            An orthorhombic unit cell with the points.
        """
        c = self.normalized()
        coordinates = c.cartesian() + gap
        shape = numpy.amax(c.vertices(), axis=0) + 2 * gap
        return UnitCell(
            Basis(
                shape,
                kind='orthorhombic',
                meta=self.meta,
            ),
            coordinates,
            self.values,
            c_basis='cartesian')

    @input_as_list
    def select(self, piece):
        """
        Selects points in this cell or grid inside a box in crystal basis.
        Images are not included.

        Args:
            piece (Iterable, ndarray): box dimensions
            `[x_from, y_from, ..., z_from, x_to, y_to, ..., z_to]`,
            where `x`, `y`, `z` are basis vectors;

        Returns:
            A numpy array with the selection mask.

        Example:
            >>> cell = UnitCell(Basis((1, 2, 3), kind="orthorhombic"), numpy.random.rand((4, 3)), numpy.arange(4))
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
        return numpy.all(
            self.coordinates < p2[numpy.newaxis, :], axis=1,
        ) & numpy.all(
            self.coordinates >= p1[numpy.newaxis, :], axis=1,
        )

    @input_as_list
    def apply(self, selection):
        """
        Applies a mask to this cell or grid to keep a subset of points.

        Args:
            selection (ndarray): a bool mask with selected species;

        Example:
            >>> cell = UnitCell(Basis((1, 2, 3), kind="orthorhombic"), numpy.random.rand((4, 3)), numpy.arange(4))
            >>> selection = cell.select((0,0,0,0.5,1,1)) # Selects species in the 'left' part of the unit cell.
            >>> cell.apply(selection) # Applies selection. Species outside the 'left' part are discarded.
        """
        selection = numpy.asanyarray(selection)
        self.coordinates = self.coordinates[selection, :]
        self.values = self.values[selection]

    @input_as_list
    def discard(self, selection):
        """
        Discards points from this cell or grid according to
        the mask specified.
        Inverse of `self.apply`.

        Args:
            selection (ndarray): species to discard;

        Example:
            >>> cell = UnitCell(Basis((1, 2, 3), kind="orthorhombic"), numpy.random.rand((4, 3)), numpy.arange(4))
            >>> selection = cell.select((0,0,0,0.5,1,1)) # Selects species in the 'left' part of the unit cell.
            >>> cell.discard(selection) # Discards selection. Species inside the 'left' part are removed.
        """
        self.apply(~numpy.asanyarray(selection))

    @input_as_list
    def cut(self, piece, select='auto'):
        """
        Selects a box inside this cell or grid and returns it as a
        smaller cell.
        Basis vectors of the resulting instance are collinear to
        those of `self`.

        Args:
            piece (Iterable, ndarray): box dimensions
            `[x_from, y_from, ..., z_from, x_to, y_to, ..., z_to]`,
            where `x`, `y`, `z` are basis vectors;
            select (ndarray, str): a custom selection mask or "auto"
            if all points in the box are to be selected;

        Returns:
            A smaller instance with a subset of points.
        """
        if select == 'auto':
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
    def merge(self, cells):
        """
        Merges points from several unit cells with the same basis.

        Args:
            cells (Iterable): cells to be merged;

        Returns:
            A new unit cell with all points merged.
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
                # Fix for not-exactly-the-same vectors
                original = c.vectors[not_d].copy()
                c.vectors[not_d] = self.vectors[not_d]
                coordinates.append(c.cartesian() + shift[numpy.newaxis, :])
                c.vectors[not_d] = original
            shift += c.vectors[d, :]
        coordinates = numpy.concatenate(coordinates, axis=0)

        return UnitCell(basis, coordinates, values, c_basis="cartesian")
    stack.__doc__ = Basis.stack.__doc__

    @input_as_list
    def supercell(self, vec):
        """
        Produces a supercell from a given unit cell.

        Args:
            vec (Iterable, ndarray): new vectors expressed
            in this basis;

        Returns:
            A supercell.

        Example:
            >>> cell = UnitCell(Basis((1, 2, 3), kind="orthorhombic"), numpy.random.rand((4, 3)), numpy.arange(4))
            >>> s_cell = cell.supercell(numpy.eye(cell.size)) # returns an exact copy
            >>> r_cell = cell.supercell(numpy.diag((1, 2, 3))) # same as cell.repeated(1, 2, 3)
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

        # Fix round-off
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
        Reduces point values in this unit cell.
        Particularly useful for counting atomic species.

        Returns:
            A dictionary containing unique point values as keys
            and numbers of their occurrences as values.
        """
        answer = {}
        for s in self.values:
            try:
                answer[s] += 1
            except KeyError:
                answer[s] = 1
        return answer

    @input_as_list
    def transpose_vectors(self, new):
        new = tuple(__xyz2i__(i) for i in new)
        Basis.transpose_vectors(self, new)
        self.coordinates = self.coordinates[:, new]
    transpose_vectors.__doc__ = Basis.transpose_vectors.__doc__

    def as_grid(self, fill=numpy.nan):
        """
        Converts this unit cell into a grid.

        Args:
            fill (object): the value to fill with undefined
            grid points' values;

        Returns:
            A grid with the data from this cell.
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
        Interpolates values between points in this cell.

        Args:
            points (Iterable, ndarray): interpolation points in crystal basis;
            driver (Callable): interpolation driver;
            periodic (bool): if True, employs the periodicity of this cell when
            interpolating;
            kwargs: driver arguments;

        Returns:
            A new unit cell with the interpolated data.
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
    A class describing data on a non-regular grid in a periodic environment.

    Args:
        vectors (Basis, ndarray): crystal basis;
        coordinates (Iterable, ndarray): list of arrays of coordinates specifying the grid;
        values (Iterable, ndarray): a multidimensional array with data on the grid;
        meta (dict): a metadata for this Grid;
        dtype (type): enforcing data type for `values`.
    """
    def __init__(self, vectors, coordinates, values, meta=None, dtype=None):

        if isinstance(vectors, Basis):
            Basis.__init__(self, vectors.vectors, meta=vectors.meta)
        else:
            Basis.__init__(self, vectors)
        if meta is not None:
            self.meta.update(meta)

        dims = self.vectors.shape[0]
        self.coordinates = list(numpy.asanyarray(c, dtype=numpy.float64) for c in coordinates)
        self.values = numpy.asanyarray(values, dtype=dtype)

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
            raise ArgumentError("The dimensionality of 'values' array is less ({:d}) than expected ({:d})".format(
                len(self.values.shape), dims))

        for i in range(dims):
            if not self.values.shape[i] == self.coordinates[i].shape[0]:
                raise ArgumentError(
                    "The dimension {:d} of 'values' array is equal to {:d} which is different from the size of the "
                    "corresponding 'coordinates' array {:d}".format(
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

    @property
    def size(self):
        """
        The number of grid points in this grid.
        """
        return numpy.prod(tuple(a.size for a in self.coordinates))

    @staticmethod
    def combine_arrays(arrays):
        """
        Transforms N 1D arrays of coordinates into (N+1)D mesh array.

        Args:
            arrays (Iterable): 1D arrays;

        Returns:
            A mesh tensor with coordinates.
        """
        mg = numpy.meshgrid(*arrays, indexing='ij')
        return numpy.concatenate(tuple(i[..., numpy.newaxis] for i in mg), axis=len(mg))

    @staticmethod
    def uniform(size, endpoint=False):
        """
        Samples the multidimensional 0-1 interval.

        Args:
            size (Iterable): positive integers specifying the number of
            sampling points per dimension;
            endpoint (bool): if True, the right boundary is included;

        Returns:
            A mesh tensor with coordinates.
        """
        return Grid.combine_arrays(tuple(
            numpy.linspace(0, 1, i, endpoint=endpoint) for i in size
        ))

    def explicit_coordinates(self):
        """
        Forms an (N+1)D array with grid points' coordinates.

        Returns:
            A mesh tensor with coordinates.
        """
        return Grid.combine_arrays(self.coordinates)

    def cartesian(self):
        """
        Computes cartesian coordinates of points in this grid.

        Returns:
            A mesh tensor with cartesian coordinates.
        """
        return self.transform_to_cartesian(self.explicit_coordinates())

    def normalized(self):
        """
        Moves all points to their periodic images in the 0-1 coordinate range.

        Returns:
            A new grid with the normalized data.
        """
        result = self.copy()

        for i in range(len(result.coordinates)):
            result.coordinates[i] = result.coordinates[i] % 1

        result.apply(tuple(numpy.argsort(a) for a in result.coordinates))

        return result

    @input_as_list
    def isolated(self, gaps, units="cartesian"):
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
    isolated.__doc__ = UnitCell.isolated.__doc__

    @input_as_list
    def select(self, piece):
        if not len(piece) == 2 * self.vectors.shape[0]:
            raise ArgumentError("Wrong coordinates array: expecting {:d} elements, found {:d}".format(
                2 * self.vectors.shape[0], len(piece)
            ))

        piece = numpy.reshape(piece, (2, -1))
        p1 = numpy.amin(piece, axis=0)
        p2 = numpy.amax(piece, axis=0)
        return list((c < mx) & (c >= mn) for c, mn, mx in zip(self.coordinates, p1, p2))
    select.__doc__ = UnitCell.select.__doc__

    @input_as_list
    def apply(self, selection):
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
    apply.__doc__ = UnitCell.apply.__doc__

    @input_as_list
    def discard(self, selection):
        self.apply(tuple(~numpy.asanyarray(i) for i in selection))
    discard.__doc__ = UnitCell.discard.__doc__

    @input_as_list
    def cut(self, piece):
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
    cut.__doc__ = UnitCell.cut.__doc__

    @input_as_list
    def merge(self, grids, fill=numpy.nan):
        """
        Merges points from several grids with the same basis.

        Args:
            grids (Iterable): grids to be merged;
            fill (object): the value to fill with undefined
            grid points' values;

        Returns:
            A new grid with all points merged.
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
        grids = [self] + grids
        d = __xyz2i__(vector)
        dims = self.vectors.shape[0]

        other_vectors = list(range(grids[0].vectors.shape[0]))
        del other_vectors[d]

        basis = Basis.stack(*grids, vector=vector, **kwargs)

        for g in grids:
            if not isinstance(g, Basis):
                raise ArgumentError('The object {} is not an instance of a Basis'.format(g))

        for i, g in enumerate(grids[1:]):
            if isinstance(g, Grid):
                for dim in other_vectors:
                    if not len(g.coordinates[dim]) == len(self.coordinates[dim]) or not numpy.all(
                            g.coordinates[dim] == self.coordinates[dim]):
                        raise ArgumentError(
                            "Grid coordinates in dimension {:d} of cells 0 and {:d} are different".format(dim, i))

        values = numpy.concatenate(tuple(grid.values for grid in grids if isinstance(grid, Grid)), axis=d)

        stacking_vectors_len = numpy.asanyarray(tuple((grid.vectors[d] ** 2).sum(axis=-1) ** .5 for grid in grids))
        shifts = numpy.cumsum(stacking_vectors_len)
        shifts = shifts / shifts[-1]

        # kx+b
        k = numpy.ones((len(grids), dims))
        k[:, d] = stacking_vectors_len / stacking_vectors_len.sum()
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
    stack.__doc__ = Basis.stack.__doc__

    @input_as_list
    def transpose_vectors(self, new):
        new = list(__xyz2i__(i) for i in new)
        Basis.transpose_vectors(self, new)
        self.coordinates = list(self.coordinates[n] for n in new)

        # Change values using swapaxes
        for i in range(len(new)):
            if not new[i] == i:
                self.values = self.values.swapaxes(i, new[i])
                new[new[i]] = new[i]
    transpose_vectors.__doc__ = Basis.transpose_vectors.__doc__

    def as_cell(self):
        """
        Converts this grid into a unit cell.

        Returns:
            A new `UnitCell` with all points from this grid.
        """
        c = self.explicit_coordinates()
        c = c.reshape((-1, c.shape[-1]))
        v = self.values.reshape((-1,) + self.values.shape[len(self.coordinates):])
        return UnitCell(self, c, v)

    def interpolate_to_array(self, points, driver=None, periodic=True, **kwargs):
        """
        Interpolates values between points in this grid.

        Args:
            points (Iterable, ndarray): interpolation points in crystal basis;
            driver (Callable): interpolation driver;
            periodic (bool): if True, employs the periodicity of this cell when
            interpolating;
            kwargs: driver arguments;

        Returns:
            An array with the interpolated data.
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
        Interpolates values between points in this grid.

        Args:
            points (Iterable): interpolation grid in crystal basis;
            kwargs: passed to `self.interpolate_to_array`;

        Returns:
            A grid with the interpolated data.
        """
        return Grid(self, points, self.interpolate_to_array(Grid.combine_arrays(points), **kwargs))

    def interpolate_to_cell(self, points, **kwargs):
        """
        Interpolates values between points in this grid.

        Args:
            points (Iterable, ndarray): interpolation points in crystal basis;
            kwargs: passed to `self.interpolate_to_array`;

        Returns:
            A cell with the interpolated data.
        """
        return UnitCell(self, points, self.interpolate_to_array(points, **kwargs))

    def interpolate_to_path(self, points, n, **kwargs):
        """
        Interpolates values between points in this grid.

        Args:
            points (Iterable, ndarray): path milestones;
            n (int): the total number of points in the path;
            kwargs: passed to `self.interpolate_to_array`;

        Returns:
            A cell with the interpolated data.
        """
        return self.interpolate_to_cell(self.generate_path(points, n), **kwargs)

    def tetrahedron_density(self, points, resolved=False, weights=None):
        """
        Calculate the density of points' values (states).
        Uses the tetrahedron method from PRB 49, 16223 by E. Blochl et al.
        3D space only.

        Args:
            points (Iterable, ndarray): values to calculate density at;
            resolved (bool): if True, returns a higher-dimensional tensor
            with spatially- and index-resolved density. The dimensions of
            the returned array are `self.values.shape + points.shape`;
            weights (Iterable, ndarray): assigns weights to points before
            calculating the density. Only for `resolved == False`;

        Returns:
            If `resolved == False`, returns a 1D array with the density.
            If `resolved == True`, returns the corresponding grid with
            values being the spatially-resolved density.
        """
        if not self.vectors.shape[0] == 3:
            raise ArgumentError("The tetrahedron density method is implemented only for 3D grids")

        initial = self.values
        points = numpy.asanyarray(points, dtype=numpy.float64)
        self.values = numpy.reshape(self.values, self.values.shape[:3] + (-1,))
        val_min = numpy.min(self.values, axis=(0, 1, 2))
        val_max = numpy.max(self.values, axis=(0, 1, 2))
        bottom = (val_max < points.min()).sum()
        top = (val_min < points.max()).sum()

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
