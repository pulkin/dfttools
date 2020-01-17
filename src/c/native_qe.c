#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

#include "generic-parser.h"

static char module_name[] = "native_qe";
static char module_docstring[] = "A module containing native parsing implementations of QE parsing routines";
static char qeproj_weights_docstring[] = "Retrieves projection weights as a numpy array";
static char qeproj_cell_docstring[] = "Retrieves a unit cell data";
static PyObject *native_qe_proj_weights(PyObject *self, PyObject *string_data);
static PyObject *native_qe_scf_cell(PyObject *self, PyObject *string_data);

static PyMethodDef module_methods[] = {
    {"qe_proj_weights", native_qe_proj_weights, METH_VARARGS, qeproj_weights_docstring},
    {"qe_scf_cell", native_qe_scf_cell, METH_VARARGS, qeproj_cell_docstring},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    module_name,         /* m_name */
    module_docstring,    /* m_doc */
    -1,                  /* m_size */
    module_methods,      /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
};

PyMODINIT_FUNC PyInit_native_qe(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;
    import_array();
    return m;
}

#else

PyMODINIT_FUNC initnative_qe(void)
{
    PyObject *m = Py_InitModule3(module_name, module_methods, module_docstring);
    if (m == NULL)
        return;
    import_array();
}

#endif


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

int cell_data(double **coordinates, char **values, npy_intp n, FILE *f) {

    *coordinates = malloc(sizeof(double) * 3 * n);
    *values = malloc(16 * n);
    if (!*coordinates || !*values) {
        if (*coordinates) free(*coordinates);
        if (*values) free(*values);
        return 0;
    }
    memset(*values, 0, 16 * n);

    for (int i=0; i<n; i++) {
        double *dest = (*coordinates) + 3 * i;
        if (fscanf(f, "%16s %lf %lf %lf", (*values) + 16 * i, dest, dest + 1, dest + 2) != 4 || !skip_line(f)) {
            free(*coordinates);
            free(*values);
            return 0;
        }
    }
    return 1;
}

static PyObject *native_qe_proj_weights(PyObject *self, PyObject *args) {
    
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

static PyObject *native_qe_scf_cell(PyObject *self, PyObject *args) {

    char *string_data;
    npy_intp n;
    if (!PyArg_ParseTuple(args, "sl", &string_data, &n))
        return NULL;

    FILE *f = fmemopen(string_data, strlen(string_data), "r");
    double *coordinates;
    char *values;
    if (!cell_data(&coordinates, &values, n, f)) {
        PyErr_SetString(PyExc_Exception, "Cell data is broken");
        return NULL;
    }
    fclose(f);

    npy_intp dims_npy[] = {n, 3};
    PyObject *arr_coordinates = PyArray_SimpleNewFromData(2, dims_npy, NPY_DOUBLE, coordinates);
    dims_npy[1] = 16;
    PyObject *arr_values = PyArray_SimpleNewFromData(2, dims_npy, NPY_BYTE, values);
    return Py_BuildValue("(O,O)", arr_coordinates, arr_values);
}
