#include <Python.h>
#include <numpy/arrayobject.h>
#include "qeproj.h"

static char module_docstring[] = "A module containing faster parsing implementations";
static char qeproj_weights_docstring[] = "Retrieves projection weights as a numpy array";

static PyObject *qeproj_weights(PyObject *self, PyObject *string_data);

static PyMethodDef module_methods[] = {
    {"qe_proj_weights", qeproj_weights, METH_VARARGS, qeproj_weights_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initnative(void)
{
    PyObject *m = Py_InitModule3("native", module_methods, module_docstring);
    if (m == NULL)
        return;

    import_array();
}

static PyObject *qeproj_weights(PyObject *self, PyObject *args) {
    
    char *string_data;
    if (!PyArg_ParseTuple(args, "s", &string_data))
        return NULL;
        
    FILE *f = fmemopen(string_data, strlen(string_data), "r");
    float *data;
    int dims[3];
    if (!weights(&data, dims, f)) {
        // Add exception
        return NULL;
    }
    fclose(f);
    
    npy_intp dims_npy[3];
    int i;
    for (i=0; i<3; i++) dims_npy[i] = dims[i];
    return PyArray_SimpleNewFromData(3, dims_npy, NPY_FLOAT, data);
}
