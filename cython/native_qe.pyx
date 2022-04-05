# cython: language_level=3
import cython
from libc.stdio cimport FILE, fscanf, ftell, fseek, SEEK_SET
from posix.stdio cimport fmemopen
from fastparse cimport skip, skip_line, skip_line_n, present, present_either, present_either2
import numpy as np


cdef int proj_n_bands(FILE *f):
    if not skip("k =", f):
        return -1
    cdef int result = 0
    while present_either2("==== e(", "k =", f) == 0:
        skip("==== e(",f)
        result += 1
    return result


cdef int proj_n_basis(FILE *f):
    if not skip("Calling projwave", f):
        return -1
    if not skip(":\n\n",f):
        return -1
    if not present("\n\n",f):
        return -1
    cdef int result = 0;
    while present_either2("state #","\n\n",f) == 0:
        skip("state #", f)
        result += 1
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def qe_proj_weights(unicode data):
    cdef const unsigned char[:] data_bytes = data.encode()
    cdef char* data_ptr = <char *> &data_bytes[0]
    cdef FILE* f = fmemopen(data_ptr, data_bytes.shape[0], "r")

    cdef int size_basis = proj_n_basis(f)
    cdef int size_bands = proj_n_bands(f)

    fseek(f, 0, SEEK_SET)
    if not skip("Calling projwave", f):
        raise RuntimeError("proj file corrupt")

    cdef int size_k = 0
    cdef long int pos = ftell(f)
    while present("k =", f):
        skip("k =", f)
        size_k += 1
    fseek(f, pos, SEEK_SET)

    result = np.zeros((size_k, size_bands, size_basis), dtype=float)
    cdef double[:, :, ::1] result_buffer = result

    cdef int nk, ne, state, w1, w2

    for nk in range(size_k):
        if not skip("k =", f):
            raise RuntimeError("proj file corrupt")

        for ne in range(size_bands):
            if not skip("==== e(", f):
                raise RuntimeError("proj file corrupt")
            if not skip_line(f):
                raise RuntimeError("proj file corrupt")
            if not skip("psi =", f):
                raise RuntimeError("proj file corrupt")

            while fscanf(f, "%d.%d*[#%d]+", &w1, &w2, &state) == 3:
                result_buffer[nk, ne, state-1] = 1.0*w1+1e-3*w2

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def qe_scf_cell(unicode data, const int size):
    cdef const unsigned char[:] data_bytes = data.encode()
    cdef char* data_ptr = <char *> &data_bytes[0]
    cdef FILE* f = fmemopen(data_ptr, data_bytes.shape[0], "r")
    cdef int i

    result_coordinates = np.empty((size, 3), dtype=float)
    cdef double[:, ::1] result_coordinates_ = result_coordinates

    result_values = np.zeros(size, dtype='a16')
    result_values_view = np.frombuffer(result_values, dtype='B')
    cdef unsigned char[:] result_values_view_ = result_values_view

    for i in range(size):
        fscanf(f, "%16s %lf %lf %lf", &result_values_view_[16 * i], &result_coordinates_[i, 0],
               &result_coordinates_[i, 1], &result_coordinates_[i, 2])

    return result_coordinates, result_values
