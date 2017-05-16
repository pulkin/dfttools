#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

#include "generic-parser.h"

static char module_docstring[] = "A module containing native parsing implementations of QE parsing routines";
static char qeproj_weights_docstring[] = "Retrieves projection weights as a numpy array";
static PyObject *qeproj_weights(PyObject *self, PyObject *string_data);

static PyMethodDef module_methods[] = {
    {"qe_proj_weights", qeproj_weights, METH_VARARGS, qeproj_weights_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initnative_qe(void)
{
    PyObject *m = Py_InitModule3("native_qe", module_methods, module_docstring);
    if (m == NULL)
        return;

    import_array();
}


int n_bands(FILE *f) {
    if (!skip("k =",f)) return -1;
    int result = 0;
    while (present_either2("==== e(","k =",f) == 0) {
        skip("==== e(",f);
        result++;
    }
    return result;
}

int n_basis(FILE *f) {
    if (!skip("Calling projwave",f)) return -1;
    if (!skip(":\n\n",f)) return -1;
    if (!present("\n\n",f)) return -1;
    int result = 0;
    while (present_either2("state #","\n\n",f) == 0) {
        skip("state #",f);
        result++;
    }
    return result;
}

int _weights(float **data, int basis_size, int bands_number, FILE *f) {
    
    if (!skip("Calling projwave", f)) return -1;
    
    int nk = 0;
    
    int nk_allocated = 1;
    int multiplier = bands_number*basis_size;
    *data = malloc(sizeof(float)*nk_allocated*multiplier);
    memset(*data, 0, sizeof(*data[0])*nk_allocated*multiplier);
        
    while (present("k =", f)) {
        
        if (!skip("k =",f)) return -1;
        
        if (nk == nk_allocated) {
            nk_allocated = nk_allocated * 2;
            *data = realloc(*data, sizeof(float)*nk_allocated*multiplier);
            memset((*data) + nk_allocated*multiplier/2, 0, sizeof(float)*nk_allocated*multiplier/2);
        }
        
        int ne;
                
        for (ne=0; ne<bands_number; ne++) {
            
            if (!skip("==== e(",f)) return -1;
            if (!skip_line(f)) return -1;
            if (!skip("psi =", f)) return -1;
            
            int state;
            int w1,w2;
            while (fscanf(f, "%d.%d*[#%d]+", &w1, &w2, &state) == 3) {
                (*data)[nk*multiplier + ne*basis_size + state-1] = 1.0*w1+1e-3*w2;
            }
        
        }
        
        nk++;
        
    }
    
    return nk;
    
}

int weights(float **data, int dims[3], FILE *f) {
    
    long int pos = ftell(f);
    int result_basis = n_basis(f);
    fseek(f,pos,SEEK_SET);

    if (result_basis<0) return 0;
    
    pos = ftell(f);
    int result_bands = n_bands(f);
    fseek(f,pos,SEEK_SET);

    if (result_bands<0) return 0;
    
    pos = ftell(f);
    int result = _weights(data, result_basis, result_bands, f);
    fseek(f,pos,SEEK_SET);
    
    if (result<0) {
        if (*data) free(*data);
        return 0;
    }
    
    dims[0] = result;
    dims[1] = result_bands;
    dims[2] = result_basis;
    return 1;
}

static PyObject *qeproj_weights(PyObject *self, PyObject *args) {
    
    char *string_data;
    if (!PyArg_ParseTuple(args, "s", &string_data))
        return NULL;
        
    FILE *f = fmemopen(string_data, strlen(string_data), "r");
    float *data;
    int dims[3];
    if (!weights(&data, dims, f)) {
        PyErr_SetString(PyExc_Exception, "Projection data is broken");
        return NULL;
    }
    fclose(f);
    
    npy_intp dims_npy[3];
    int i;
    for (i=0; i<3; i++) dims_npy[i] = dims[i];
    return PyArray_SimpleNewFromData(3, dims_npy, NPY_FLOAT, data);
}
