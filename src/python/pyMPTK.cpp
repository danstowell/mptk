#include "pyMPTK.h"

static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initmptk(void)
{
    PyObject* m;

    if (PyType_Ready(&bookType) < 0)
        return;

    m = Py_InitModule3("mptk", module_methods,
                       "MPTK - Matching Pursuit ToolKit - decompose and analyse signals.");

    if (m == NULL)
      return;

    Py_INCREF(&bookType);
    PyModule_AddObject(m, "book", (PyObject *)&bookType);
}
