import numpy
cimport numpy, cython
from numericalunits import eV

#@cython.boundscheck(True)
def tetrahedron(cell, numpy.ndarray[numpy.double_t, ndim=1] pts_at):
    
    cdef double vol = cell.volume()
    cdef int i,j,k,i1,j1,k1,v1,v2,v3,v4,b,n
    cdef double v_t, e1, e2, e3, e4, e, _t
    
    cdef numpy.ndarray[numpy.double_t, ndim=4] crds = cell.cartesian()
    cdef numpy.ndarray[numpy.double_t, ndim=4] vals = cell.values
    cdef numpy.ndarray[numpy.double_t, ndim=2] cc = numpy.zeros((8,3), dtype = numpy.double)
    cdef numpy.ndarray[numpy.double_t, ndim=2] vv = numpy.zeros((8,cell.values.shape[-1]), dtype = numpy.double)

    #cdef numpy.ndarray[numpy.double_t, ndim=1] result = numpy.zeros(pts_at.shape[0], dtype = numpy.double)
    cdef numpy.ndarray[numpy.double_t, ndim=5] result = numpy.zeros((
            vals.shape[0],
            vals.shape[1],
            vals.shape[2],
            vals.shape[3],
            pts_at.shape[0],
        ), dtype = numpy.double)
        
    # Parallelipiped loop
    for i in range(crds.shape[0]):
        i1 = (i+1) % crds.shape[0]
        for j in range(crds.shape[1]):
            j1 = (j+1) % crds.shape[1]
            for k in range(crds.shape[2]):
                k1 = (k+1) % crds.shape[2]
                
                cc = crds[
                    ( i, i, i, i,i1,i1,i1,i1),
                    ( j, j,j1,j1, j, j,j1,j1),
                    ( k,k1, k,k1, k,k1, k,k1),
                ]
                    
                if i1 == 0:
                    cc[4:] += cell.vectors[0,numpy.newaxis,:]
                if j1 == 0:
                    cc[2:4] += cell.vectors[1,numpy.newaxis,:]
                    cc[6:] += cell.vectors[1,numpy.newaxis,:]
                if k1 == 0:
                    cc[1::2] += cell.vectors[2,numpy.newaxis,:]
               
                vv = vals[
                    ( i, i, i, i,i1,i1,i1,i1),
                    ( j, j,j1,j1, j, j,j1,j1),
                    ( k,k1, k,k1, k,k1, k,k1),
                ]
                
                # Tetrahedron loop
                for v1,v2,v3,v4 in (
                    (0,1,2,5),
                    (1,2,3,5),
                    (0,2,4,5),
                    (2,4,5,6),
                    (2,3,5,7),  
                    (2,5,6,7),
                ):
                    
                    v_t = abs(numpy.linalg.det((cc[v1]-cc[v4],cc[v2]-cc[v4],cc[v3]-cc[v4])))/vol/6
                    
                    # Band loop
                    for b in range(vals.shape[3]):
                    
                        e1, e2, e3, e4 = sorted((vv[v1,b],vv[v2,b],vv[v3,b],vv[v4,b]))
                            
                        for n in range(pts_at.shape[0]):
                            e = pts_at[n]
        
                            if e>e1 and e<=e2:
                                _t = v_t * 3 * (e-e1)**2 / (e2-e1) / (e3-e1) / (e4-e1)
                            elif e>e2 and e<=e3:
                                _t = v_t / (e3-e1) / (e4-e1) * (
                                    3 * (e2-e1) + 6 * (e-e2) - 3 * (e4 + e3 - e2 - e1) * (e - e2)**2 / (e3 - e2) / (e4 - e2)
                                )
                            elif e>e3 and e<e4:
                                _t = v_t * 3 * (e4-e)**2 / (e4-e1) / (e4-e2) / (e4-e3)
                            else:
                                _t = 0
                                
                            result[i,j,k,b,n] += _t
    return result
