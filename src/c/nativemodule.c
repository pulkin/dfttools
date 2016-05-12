#include <Python.h>
#include <numpy/arrayobject.h>
#include "qeproj.h"
#include "openmx-hks.h"

// OpenMX-HKS data
#define OPENMX_HKS_MAX_DATASETS 16
struct basis_description *openmx_hks_buffer[OPENMX_HKS_MAX_DATASETS];

static char module_docstring[] = "A module containing native parsing implementations";
static char qeproj_weights_docstring[] = "Retrieves projection weights as a numpy array";
static char openmx_hks_load_docstring[] = "Loads an HKS file";
static char openmx_hks_unload_docstring[] = "Unloads an hks file";
static char openmx_hks_blocks_docstring[] = "Retrieves tight-binding blocks from the hks file";
static char openmx_hks_basis_docstring[] = "Retrieves the atomic basis of an HKS";
static char openmx_hks_slice_basis_docstring[] = "Slices the basis";

static PyObject *qeproj_weights(PyObject *self, PyObject *string_data);
static PyObject *openmx_hks_load(PyObject *self, PyObject *args);
static PyObject *openmx_hks_unload(PyObject *self, PyObject *args);
static PyObject *openmx_hks_blocks(PyObject *self, PyObject *args);
static PyObject *openmx_hks_basis(PyObject *self, PyObject *args);
static PyObject *openmx_hks_slice_basis(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"qe_proj_weights", qeproj_weights, METH_VARARGS, qeproj_weights_docstring},
    {"openmx_hks_load", openmx_hks_load, METH_VARARGS, openmx_hks_load_docstring},
    {"openmx_hks_unload", openmx_hks_unload, METH_VARARGS, openmx_hks_unload_docstring},
    {"openmx_hks_blocks", openmx_hks_blocks, METH_VARARGS, openmx_hks_blocks_docstring},
    {"openmx_hks_basis", openmx_hks_basis, METH_VARARGS, openmx_hks_basis_docstring},
    {"openmx_hks_slice_basis", openmx_hks_slice_basis, METH_VARARGS, openmx_hks_slice_basis_docstring},
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

static PyObject *openmx_hks_load(PyObject *self, PyObject *args) {
    
    // Parse input
    PyObject *py_file;
    if (!PyArg_ParseTuple(args, "O", &py_file)) {
        PyErr_SetString(PyExc_Exception, "File is expected");
        return NULL;
    }
    
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
    
    // Store in buffer
    int i;
    for (i=0; i<OPENMX_HKS_MAX_DATASETS; i++) {
        if (!openmx_hks_buffer[i]) {
            
            struct hks_data *data_ = (struct hks_data*)malloc(sizeof(struct hks_data));
            data_[0] = data;
            openmx_hks_buffer[i] = (struct basis_description*)malloc(sizeof(struct basis_description));
            
            // Make a basis
            make_basis(data_, openmx_hks_buffer[i]);
            
            return PyInt_FromLong(i);
        }
    }
    
    PyErr_SetString(PyExc_IOError, "The HKS buffer is full");
    dispose_hks(&data);
    return NULL;
}

static PyObject *openmx_hks_unload(PyObject *self, PyObject *args) {
    
    // Parse input
    int handle;
    if (!PyArg_ParseTuple(args, "i", &handle)) {
        PyErr_SetString(PyExc_Exception, "Integer handle is expected");
        return NULL;
    }
    
    // Check if available and unload
    if (openmx_hks_buffer[handle]) {
        
        dispose_basis(openmx_hks_buffer[handle]);
        dispose_hks(openmx_hks_buffer[handle]->data);
        free(openmx_hks_buffer[handle]->data);
        free(openmx_hks_buffer[handle]);
        
    }
    
    return Py_None;
}

static PyObject *openmx_hks_blocks(PyObject *self, PyObject *args) {
    
    // Parse input
    int handle;
    if (!PyArg_ParseTuple(args, "i", &handle)) {
        PyErr_SetString(PyExc_Exception, "Integer handle is expected");
        return NULL;
    }
    
    // Check if available
    struct basis_description *basis = openmx_hks_buffer[handle];
    
    if (!basis) {
        PyErr_SetString(PyExc_Exception, "No HKS data found under this handle");
        return NULL;
    }
    
    // Calculate blocks
    PyObject *result = PyList_New(basis->data->cell_replica_number);
    int i,j,x,y,z;
    
    for (i=0; i<basis->data->cell_replica_number; i++) {
        
        PyObject *block = PyList_New(5);
        
        for (j=0; j<3; j++) PyList_SetItem(block, j, PyInt_FromLong(basis->data->cell_replicas[i].index[j]));
        x = basis->data->cell_replicas[i].index[0];
        y = basis->data->cell_replicas[i].index[1];
        z = basis->data->cell_replicas[i].index[2];
        
        // Read block data
        if (calculate_block(basis[0], x, y, z, NULL, NULL)) {
            
            npy_intp array_size[2] = {basis->size, basis->size};
            
            PyArrayObject *H = PyArray_SimpleNew(2, array_size, NPY_CDOUBLE);
            PyArrayObject *S = PyArray_SimpleNew(2, array_size, NPY_CDOUBLE);
            
            // Unsafe variant
            calculate_block(basis[0], x, y, z, (struct F_complex*)(H->data), (struct F_complex*)(S->data));
            
            // Safe variant
            //struct F_complex H_[basis->size*basis->size];
            //struct F_complex S_[basis->size*basis->size];
            //calculate_block(basis[0], x, y, z, H_, S_);
            //int k,l;
            //for (k=0; k<basis->size; k++) for (l=0; l<basis->size; l++) {
                //struct F_complex *v_;
                //v_ = (struct F_complex *)PyArray_GETPTR2(H,k,l);
                //v_->r = H_[k*basis->size+l].r;
                //v_->i = H_[k*basis->size+l].i;
                //v_ = (struct F_complex *)PyArray_GETPTR2(S,k,l);
                //v_->r = S_[k*basis->size+l].r;
                //v_->i = S_[k*basis->size+l].i;
            //}
            
            PyList_SetItem(block, 3, (PyObject*)H);
            PyList_SetItem(block, 4, (PyObject*)S);
        
        } else {
            
            PyList_SetItem(block, 3, Py_None);
            PyList_SetItem(block, 4, Py_None);
            
        }
        
        if (PyList_SetItem(result, i, block) != 0) {
            PyErr_SetString(PyExc_Exception, "Internal error 1");
            return NULL;
        }
        
    }
    
    return result;
}

static PyObject *openmx_hks_basis(PyObject *self, PyObject *args) {
    
    // Parse input
    int handle;
    if (!PyArg_ParseTuple(args, "i", &handle)) {
        PyErr_SetString(PyExc_Exception, "Integer handle is expected");
        return NULL;
    }
    
    // Check if available
    struct basis_description *basis = openmx_hks_buffer[handle];
    
    if (!basis) {
        PyErr_SetString(PyExc_Exception, "No HKS data found under this handle");
        return NULL;
    }

    npy_intp array_size[2] = {basis->size, 3};
    PyArrayObject *basis_npy = PyArray_SimpleNew(2, array_size, NPY_INT);
    
    // Unsafe variant
    memcpy(basis_npy->data, basis->r2s, sizeof(struct plain_index)*basis->size);
    
    // Safe variant
    //int i;
    //for (i=0; i<basis->size; i++) {
        
        //int *x = (int*) PyArray_GETPTR2(basis_npy,i,0);
        //*x = basis->r2s[i].spin;
        
        //x = (int*) PyArray_GETPTR2(basis_npy,i,1);
        //*x = basis->r2s[i].atom;
        
        //x = (int*) PyArray_GETPTR2(basis_npy,i,2);
        //*x = basis->r2s[i].orbital;
        
    //}
    
    return (PyObject*)basis_npy;
}

static PyObject *openmx_hks_slice_basis(PyObject *self, PyObject *args) {
    
    // Parse input
    int handle;
    PyArrayObject *numpy_array;
    if (!PyArg_ParseTuple(args, "iO!", &handle, &PyArray_Type, &numpy_array)) {
        PyErr_SetString(PyExc_Exception, "Integer handle and an array are expected");
        return NULL;
    }
    
    // Check if available
    struct basis_description *basis = openmx_hks_buffer[handle];
    
    if (!basis) {
        PyErr_SetString(PyExc_Exception, "No HKS data found under this handle");
        return NULL;
    }

    // Slice
    int *slice = (int*) numpy_array->data;
    int i;
    for (i=0; i<basis->size; i++) if (slice[i] != 0 && slice[i] != 1) {
        PyErr_SetString(PyExc_Exception, "Invalid array passed");
        return NULL;
    }
    slice_basis(basis, slice);
    return Py_None;
}
