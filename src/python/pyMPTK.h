#ifndef PYMPTK_H_INCLUDED
#define PYMPTK_H_INCLUDED

#include <Python.h>
#include <structmember.h>
#include "numpy/arrayobject.h"

#define HAVE_FFTW3 1

#include "mptk.h"

//////////////////////////////////////////////////////////
// declarations re "book":

typedef struct {
	PyObject_HEAD
	MP_Book_c *book;
	int numAtoms;
	int numChans;
	int numSamples;
	int sampleRate;
	PyObject *atomType;
	PyObject *atomParams;
} book;

void       book_dealloc(book* self);
PyObject * book_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
int        book_init(book *self, PyObject *args, PyObject *kwds);
PyObject * book_read(book* self, PyObject *args);
PyObject * book_short_info(book* self);

static PyMethodDef book_methods[] = {
	{"read", (PyCFunction)book_read, METH_VARARGS,
		"read a book file"
		},
	{"short_info", (PyCFunction)book_short_info, METH_NOARGS,
		"print short info"
		},
	{NULL}  /* Sentinel */
};

static PyMemberDef book_members[] = {
	{"numAtoms", T_INT, offsetof(book, numAtoms), 0,
		"number of atoms"},
	{"numChans", T_INT, offsetof(book, numChans), 0,
		"number of chanels"},
	{"numSamples", T_INT, offsetof(book, numSamples), 0,
		"number of samples"},
	{"sampleRate", T_INT, offsetof(book, sampleRate), 0,
		"sample rate"},
	{"atomType", T_OBJECT_EX, offsetof(book, atomType), 0,
		"atom type"},
	{"atomParams", T_OBJECT_EX, offsetof(book, atomParams), 0,
		"atom parameters"},
	{NULL}  /* Sentinel */
};

static PyTypeObject bookType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "pyMPTK.book",             /*tp_name*/
    sizeof(book),             /*tp_basicsize*/
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

////////////////////////////////////////////////////
// module methods

PyObject * mptk_loadconfig(PyObject *self, PyObject *args);

PyObject * mptk_decompose(PyObject *self, PyObject *args, PyObject *keywds);

struct mptk_decompose_result { PyArrayObject* residual; };
int mptk_decompose_body(const PyArrayObject *numpysignal, const char *dictpath, const int samplerate, const unsigned long int numiters, const char *method, const bool getdecay, mptk_decompose_result& result);

MPTK_LIB_EXPORT extern PyArrayObject* mp_create_numpyarray_from_signal(MP_Signal_c *signal);
MPTK_LIB_EXPORT extern MP_Signal_c*   mp_create_signal_from_numpyarray(const PyArrayObject *nparray);

#endif

