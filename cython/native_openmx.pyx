# cython: language_level=3
import cython
from libc.stdio cimport FILE, fscanf
from posix.stdio cimport fmemopen
from fastparse cimport skip, skip_line, skip_line_n, present, present_either
import numpy as np


def openmx_bands_bands(unicode data):
    cdef const unsigned char[:] data_bytes = data.encode()
    cdef char* data_ptr = <char *> &data_bytes[0]
    cdef FILE* f = fmemopen(data_ptr, data_bytes.shape[0], "r")

    cdef int nbands
    if fscanf(f, "%d", &nbands) != 1:
        raise RuntimeError("openmx bands file corrupt")
    if not skip_line_n(f, 2):
        raise RuntimeError("openmx bands file corrupt")
    nbands += 3

    cdef int npath
    if fscanf(f, "%d", &npath) != 1:
        raise RuntimeError("openmx bands file corrupt")

    cdef int i, j, nk = 0, nk_add
    for i in range(npath):
        if fscanf(f, "%d", &nk_add) != 1:
            raise RuntimeError("openmx bands file corrupt")
        nk += nk_add
        if not skip_line(f):
            raise RuntimeError("openmx bands file corrupt")

    result = np.zeros((nk, nbands), dtype=float)
    cdef double[:, ::1] result_buffer = result
    for i in range(nk):
        if fscanf(f, "%d", &nk_add) != 1:
            raise RuntimeError("openmx bands file corrupt")

        for j in range(nbands):
            if fscanf(f, "%lf", &result_buffer[i, j]) != 1:
                raise RuntimeError("openmx bands file corrupt")
    return result
