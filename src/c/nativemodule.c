#include <Python.h>
#include <numpy/arrayobject.h>
#include "qeproj.h"
#include "openmx-hks.h"

static char module_docstring[] = "A module containing native parsing implementations";
static char qeproj_weights_docstring[] = "Retrieves projection weights as a numpy array";
static char openmx_hks_blocks_docstring[] = "Retrieves tight-binding blocks from the hks file";

static PyObject *qeproj_weights(PyObject *self, PyObject *string_data);
static PyObject *openmx_hks_blocks(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"qe_proj_weights", qeproj_weights, METH_VARARGS, qeproj_weights_docstring},
    {"openmx_hks_blocks", openmx_hks_blocks, METH_VARARGS, openmx_hks_blocks_docstring},
    {NULL, NULL, 0, NULL}
};

#define PRINT(x) printf("%s\n",PyString_AsString(PyObject_Repr(x)))

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

static PyObject *openmx_hks_blocks(PyObject *self, PyObject *args) {
    
    // Parse output
    PyObject *py_file;
    if (!PyArg_ParseTuple(args, "O", &py_file))
        return NULL;
    
    // Get file handle
    FILE *file = PyFile_AsFile(py_file);
    if (file == NULL) {
        PyErr_SetString(PyExc_IOError, "An HKS file is NULL");
        return NULL;
    }
    rewind(file);
    
    // Read HKS
    struct hks_data data;
    int read_result;
    if ((read_result = read_hks(file,&data)) != SUCCESS) {
        switch (read_result) {
            case ERR_FILE_IO: PyErr_SetString(PyExc_IOError, "An IO error occurred while reading HKS file"); break;
            case ERR_VERSION: PyErr_SetString(PyExc_IOError, "Unrecognized version of the HKS file"); break;
            case ERR_FILE_STRUCTURE: PyErr_SetString(PyExc_IOError, "HKS file is broken"); break;
        }
        dispose_hks(&data);
        return NULL;
    }
    
    // Make a basis
    struct basis_description basis;
    make_basis(&data, &basis);
    
    // Calculate blocks
    PyObject *result = PyList_New(data.cell_replica_number);
    int i,j,x,y,z;
    
    for (i=0; i<data.cell_replica_number; i++) {
        
        PyObject *block = PyList_New(5);
        
        for (j=0; j<3; j++) PyList_SetItem(block, j, PyInt_FromLong(data.cell_replicas[i].index[j]));
        x = data.cell_replicas[i].index[0];
        y = data.cell_replicas[i].index[1];
        z = data.cell_replicas[i].index[2];
        
        // Read block data
        if (calculate_block(basis, x, y, z, NULL, NULL)) {
            
            npy_intp array_size[2] = {basis.size, basis.size};
            
            PyObject *H = PyArray_SimpleNew(2, array_size, NPY_CDOUBLE);
            PyObject *S = PyArray_SimpleNew(2, array_size, NPY_CDOUBLE);
            
            // Unsafe variant
            //calculate_block(basis, x, y, z, (struct F_complex*)PyArray_GETPTR2(H,0,0), (struct F_complex*)PyArray_GETPTR2(S,0,0));
            
            // Safe variant
            struct F_complex H_[basis.size*basis.size];
            struct F_complex S_[basis.size*basis.size];
            calculate_block(basis, x, y, z, H_, S_);
            int k,l;
            for (k=0; k<basis.size; k++) for (l=0; l<basis.size; l++) {
                struct F_complex *v_;
                v_ = (struct F_complex *)PyArray_GETPTR2(H,k,l);
                v_->r = H_[k*basis.size+l].r;
                v_->i = H_[k*basis.size+l].i;
                v_ = (struct F_complex *)PyArray_GETPTR2(S,k,l);
                v_->r = S_[k*basis.size+l].r;
                v_->i = S_[k*basis.size+l].i;
            }
            
            PyList_SetItem(block, 3, H);
            PyList_SetItem(block, 4, S);
        
        } else {
            
            PyList_SetItem(block, 3, Py_None);
            PyList_SetItem(block, 4, Py_None);
            
        }
        
        if (PyList_SetItem(result, i, block) != 0) {
            PyErr_SetString(PyExc_Exception, "Internal error 1");
            return NULL;
        }
        
    }
    
    // Dispose
    dispose_basis(&basis);
    dispose_hks(&data);
    
    return result;
}
