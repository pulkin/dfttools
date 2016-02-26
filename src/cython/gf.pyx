import numpy
from scipy import linalg

cimport numpy, cython

#@cython.boundscheck(True)
def greens_function(
    numpy.complex128_t E,
    numpy.ndarray[numpy.complex128_t, ndim=2] H0,
    numpy.ndarray[numpy.complex128_t, ndim=2] H1,
    numpy.ndarray[numpy.complex128_t, ndim=2] S0,
    numpy.ndarray[numpy.complex128_t, ndim=2] S1,
    double tolerance,
    int maxiter,
):
    
    cdef numpy.ndarray[numpy.complex_t, ndim=2] es0 = E*S0 - H0
    cdef numpy.ndarray[numpy.complex_t, ndim=2] e00 = E*S0 - H0
    cdef numpy.ndarray[numpy.complex_t, ndim=2] alp = -E*S1  + H1
    cdef numpy.ndarray[numpy.complex_t, ndim=2] bet = -E*S1.conj().T + H1.conj().T
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
            
    return gr00
