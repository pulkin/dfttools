#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

#include "generic-parser.h"

static char module_docstring[] = "A module containing native parsing implementations of OpenMX parsing routines";
static char openmx_bands_bands_docstring[] = "Retrieves OpenMX bands data";
static PyObject *openmx_bands_bands(PyObject *self, PyObject *string_data);

static PyMethodDef module_methods[] = {
    {"openmx_bands_bands", openmx_bands_bands, METH_VARARGS, openmx_bands_bands_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initnative_openmx(void)
{
    PyObject *m = Py_InitModule3("native_openmx", module_methods, module_docstring);
    if (m == NULL)
        return;

    import_array();
}

int bands(double **data, int dims[2], FILE *f) {

    int nbands;
    if (!(fscanf(f, "%d", &nbands) == 1)) return 0;
    if (!skip_line_n(f,2)) return 0;
    nbands += 3;
    
    int npath;
    if (!(fscanf(f, "%d", &npath) == 1)) return 0;
    
    int i;
    int nk = 0;
    int nk_add;
    for (i=0; i<npath; i++) {
        if (!(fscanf(f, "%d", &nk_add) == 1)) return 0;
        nk += nk_add;
        if (!skip_line(f)) return 0;
    }
    
    *data = malloc(sizeof(double) * nk * nbands);
    if (*data == NULL) return 0;
    for (i=0; i<nk; i++) {
        
        if (!(fscanf(f, "%d", &nk_add) == 1)) {
            free(*data);
            return 0;
        }
        
        int j;
        
        for (j=0; j<nbands; j++) if (!(fscanf(f, "%lf", (*data) + i*nbands + j) == 1)) {
            free(*data);
            return 0;
        }
    }
    dims[0] = nk;
    dims[1] = nbands;
    return 1;
}

static PyObject *openmx_bands_bands(PyObject *self, PyObject *args) {
    
    char *string_data;
    if (!PyArg_ParseTuple(args, "s", &string_data))
        return NULL;
        
    FILE *f = fmemopen(string_data, strlen(string_data), "r");
    double *data;
    int dims[2];
    if (!bands(&data, dims, f)) {
        PyErr_SetString(PyExc_Exception, "Bands data is broken");
        return NULL;
    }
    fclose(f);
    
    npy_intp dims_npy[2];
    int i;
    for (i=0; i<2; i++) dims_npy[i] = dims[i];
    return PyArray_SimpleNewFromData(2, dims_npy, NPY_DOUBLE, data);
}
