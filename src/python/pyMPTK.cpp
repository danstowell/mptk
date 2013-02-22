// NB this define must come before the includes -- see http://mail.scipy.org/pipermail/numpy-discussion/2001-July/013060.html
#define PY_ARRAY_UNIQUE_SYMBOL Py_Array_API_pymptk
#include "pyMPTK.h"

static PyMethodDef module_methods[] = {
	{"loadconfig", mptk_loadconfig, METH_VARARGS, "load MPTK config file from a specific path. do this BEFORE running decompose() etc."},
	{"decompose" , mptk_decompose,  METH_VARARGS, "decompose a signal into a 'book' and residual, using Matching Pursuit or related methods."},
	{NULL}  /* Sentinel */
};

PyObject *
mptk_loadconfig(PyObject *self, PyObject *args)
{
	const char *cfgpath;
	if (!PyArg_ParseTuple(args, "s:mptk_loadconfig", &cfgpath))
		return NULL;
	printf("mptk_loadconfig: parsed args\n");
	int result = MPTK_Env_c::get_env()->load_environment(cfgpath);
	printf("mptk_loadconfig: done, result %i, about to return\n", result);
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
mptk_decompose(PyObject *self, PyObject *args)
{
	// book, residual, decay = mptk.decompose(sig, dictpath, samplerate, [ numiters=10, method='mp', getdecay=False ])
	PyObject *pysignal; // note: do not touch the referencecount for this
	PyArrayObject *numpysignal;
	const char *dictpath;
	int samplerate;
	unsigned long int numiters=10;
	const char *method="mp";
	int getdecay=0;
	if (!PyArg_ParseTuple(args, "Osi|ksi:mptk_decompose", &pysignal, &dictpath, &samplerate, &numiters, &method, &getdecay))
		return NULL;
	printf("mptk_decompose: parsed args\n");

	// Now to get a usable numpy array from the opaque obj
	// TODO: currently can only handle arrays with dtype=float32. Is there a way to automatically convert if needed?
	numpysignal = (PyArrayObject*) PyArray_ContiguousFromObject(pysignal, PyArray_FLOAT, 2, 2); // 1D or 2D
	if(numpysignal==NULL){
		printf("mptk_decompose failed to get numpy array object\n"); // todo: proper error
		return NULL;
	}
	// From this point on: remember to do Py_DECREF(numpysignal) if terminating early

	// Here's where we call the heavy stuff
	mptk_decompose_result result;
	int intresult = mptk_decompose_body(numpysignal, dictpath, samplerate, numiters, method, getdecay==0, result);

	printf("mptk_decompose: about to return\n");
	Py_DECREF(numpysignal); // destroy the contig array
	return Py_BuildValue("iOi", 37, result.residual, 31); // LATER return (book, residual, decay)
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
    printf("Py_InitModule3(mptk) OK, and done import_array()\n");
}
