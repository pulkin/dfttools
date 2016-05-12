import numpy
from scipy import linalg

cimport numpy, cython

#@cython.boundscheck(True)
def greens_function(
    numpy.ndarray[numpy.complex128_t, ndim=2] W0,
    numpy.ndarray[numpy.complex128_t, ndim=2] W_positive,
    numpy.ndarray[numpy.complex128_t, ndim=2] W_negative,
    double tolerance,
    int maxiter,
):
    
    cdef numpy.ndarray[numpy.complex_t, ndim=2] es0 = W0
    cdef numpy.ndarray[numpy.complex_t, ndim=2] e00 = W0
    cdef numpy.ndarray[numpy.complex_t, ndim=2] alp = -W_positive
    cdef numpy.ndarray[numpy.complex_t, ndim=2] bet = -W_negative
    cdef numpy.ndarray[numpy.complex_t, ndim=2] gr00 = linalg.inv(es0)
    cdef numpy.ndarray[numpy.complex_t, ndim=2] gt = gr00
    
    cdef numpy.ndarray[numpy.complex_t, ndim=2] gr01
    cdef numpy.ndarray[numpy.complex_t, ndim=2] gr02
    
    cdef double rms
    
    cdef int iterations = 0
    
    while True:
        gr02 = linalg.inv(e00)
        gr01 = numpy.dot(gr02, bet)
        gr00 = numpy.dot(alp, gr01)
        es0 = es0 - gr00
        gr01 = numpy.dot(gr02, alp)
        gr00 = numpy.dot(gr02, bet)
        gr02 = numpy.dot(bet, gr01)
        e00 = e00 - gr02
        gr02 = numpy.dot(alp, gr00)
        e00 = e00 - gr02
        gr02 = numpy.dot(alp, gr01)
        alp = gr02
        gr02 = numpy.dot(bet, gr00)
        bet = gr02
        gr00 = linalg.inv(es0)
        rms = numpy.abs(gt-gr00).max()
        iterations += 1
        if rms>tolerance and iterations<maxiter:
            gt = gr00
        else:
            break
    
    if rms>tolerance:
        raise Exception("Green's function iteration error: after {:d} iterations the error is {:e} (required {:e})".format(iterations, rms, tolerance))
        
    return gr00
