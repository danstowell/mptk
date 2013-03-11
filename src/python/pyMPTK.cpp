// NB this define must come before the includes -- see http://mail.scipy.org/pipermail/numpy-discussion/2001-July/013060.html
#define PY_ARRAY_UNIQUE_SYMBOL Py_Array_API_pymptk
#include "pyMPTK.h"

/**
* Author: Dan Stowell
* Description: This file defines the initialisation for the module etc, as well as the bits which get data in/out of PyArray format.
*/

static PyMethodDef module_methods[] = {
	{"loadconfig", mptk_loadconfig, METH_VARARGS, "load MPTK config file from a specific path. do this BEFORE running decompose() etc."},
	{"decompose" , (PyCFunction) mptk_decompose,  METH_VARARGS | METH_KEYWORDS, "decompose a signal into a 'book' and residual, using Matching Pursuit or related methods.\n(book, residual, decay) = mptk.decompose(sig, dictpath, samplerate, [ snr=0.5, numiters=10, method='mp', getdecay=False, ... ])"},
	{NULL}  /* Sentinel */
};

PyTypeObject bookType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "mptk.book",               /*tp_name*/
    sizeof(BookObject),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)book_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "book objects",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    book_methods,             /* tp_methods */
    book_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)book_init,      /* tp_init */
    0,                         /* tp_alloc */
    book_new,                 /* tp_new */
};

PyObject *
mptk_loadconfig(PyObject *self, PyObject *args)
{
	const char *cfgpath;
	if (!PyArg_ParseTuple(args, "s:mptk_loadconfig", &cfgpath))
		return NULL;
	//printf("mptk_loadconfig: parsed args\n");
	int result = MPTK_Env_c::get_env()->load_environment(cfgpath);
	//printf("mptk_loadconfig: done, result %i, about to return\n", result);
	return Py_BuildValue("i", result);
}


// This method needs to stay in the same file as initmptk(), because of the use of PyArray_FromDims() interacts with import_array()
PyArrayObject* mp_create_numpyarray_from_signal(MP_Signal_c *signal){
	unsigned long int nspls = signal->numSamples;
	unsigned      int nchans= signal->numChans;
	printf("mp_create_numpyarray_from_signal(): %i samples, %i channels\n", nspls, nchans);
	int dims[2];
	dims[0] = nspls;
	dims[1] = nchans;
	PyArrayObject* nparray = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_FLOAT);
	float *signal_data = (float *)nparray->data;
	// There may be a faster way. Can probably memcpy each individual channel, though possibly not the whole block
	for (unsigned int channel=0; channel < nchans; ++channel) {
		for (unsigned long int sample=0; sample < nspls; ++sample) {
			signal_data[channel*nspls + sample] = signal->channel[channel][sample];
		}
	}
	return nparray;
}



// This method needs to stay in the same file as initmptk(), because of the use of import_array()
PyObject *
mptk_decompose(PyObject *self, PyObject *args, PyObject *keywds)
{
	// book, residual, decay = mptk.decompose(sig, dictpath, samplerate, [ snr=0.5, numiters=10, ... ])
	PyObject *pysignal; // note: do not touch the referencecount for this
	PyArrayObject *numpysignal;
	const char *dictpath;
	float samplerate;
	unsigned long int numiters=0; // 0 is flag to use SNR not numiters
	float snr=0.5f;
	const char *method="mp";
	int getdecay=0;
	const char *bookpath="";
	static char *kwlist[] = {"signal", "dictpath", "samplerate", "numiters", "snr", "method", "getdecay", "bookpath", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "Osf|ksis", kwlist,
		&pysignal, &dictpath, &samplerate,
		&numiters, &snr, &method, &getdecay, &bookpath))
		return NULL;
	//printf("mptk_decompose: parsed args\n");

	// Now to get a usable numpy array from the opaque obj
	// TODO: currently can only handle arrays with dtype=float32. Is there a way to automatically convert if needed?
	numpysignal = (PyArrayObject*) PyArray_ContiguousFromObject(pysignal, PyArray_FLOAT, 1, 2); // 1D or 2D
	if(numpysignal==NULL){
		printf("mptk_decompose failed to get numpy array object\n"); // todo: proper error
		return NULL;
	}
	// From this point on: remember to do Py_DECREF(numpysignal) if terminating early

	// Here's where we call the heavy stuff
	mptk_decompose_result result;
	int intresult = mptk_decompose_body(numpysignal, dictpath, (int)samplerate, numiters, snr, method, getdecay==0, bookpath, result);

	//printf("mptk_decompose: about to return\n");
	Py_DECREF(numpysignal); // destroy the contig array
	return Py_BuildValue("OOO", result.thebook, result.residual, Py_None); // TODO: return decay as third item
}



#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initmptk(void)
{
    PyObject* m;

    if (PyType_Ready(&bookType) < 0){
        printf("not PyType_Ready(&bookType)\n");
        return;
   }

    m = Py_InitModule3("mptk", module_methods,
                       "MPTK - Matching Pursuit ToolKit - decompose and analyse signals.");

    if (m == NULL){
        printf("Py_InitModule3(mptk) failed\n");
      return;
    }

    Py_INCREF(&bookType);
    PyModule_AddObject(m, "book", (PyObject *)&bookType);

    import_array(); // numpy init -- must be called after InitModule
    //printf("Py_InitModule3(mptk) OK, and done import_array()\n");
}
