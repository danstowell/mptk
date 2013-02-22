#include "pyMPTK.h"

static PyMethodDef module_methods[] = {
	{"decompose", mptk_decompose, METH_VARARGS, "decompose a signal into a 'book' and residual, using Matching Pursuit or related methods."},
	{NULL}  /* Sentinel */
};

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

	// Here's where we would do the heavy stuff (later)
	int result = mptk_decompose_body(numpysignal, dictpath, samplerate, numiters, method, getdecay==0);

	printf("mptk_decompose: about to return\n");
	Py_DECREF(numpysignal); // destroy the contig array
	return Py_BuildValue("iii", 37, 34, 31); // LATER return (book, residual, decay)
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
