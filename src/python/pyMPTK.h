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
	MP_Book_c *mpbook;
	int numChans;
	int numSamples;
	int sampleRate;
	PyObject *atoms;
} BookObject;

static PyMemberDef book_members[] = {
	{"numChans",   T_INT,       offsetof(BookObject, numChans),   0, "number of chanels"},
	{"numSamples", T_INT,       offsetof(BookObject, numSamples), 0, "number of samples"},
	{"sampleRate", T_INT,       offsetof(BookObject, sampleRate), 0, "sample rate"},
	{"atoms",      T_OBJECT_EX, offsetof(BookObject, atoms),      0, "list of atoms (each a dict)"},
	{NULL}  /* Sentinel */
};

void       book_dealloc(BookObject* self);
PyObject * book_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
int        book_init(BookObject *self, PyObject *args, PyObject *kwds);
PyObject * book_read(BookObject* self, PyObject *args);
int pybook_from_mpbook(BookObject* self, MP_Book_c *mpbook); // NB BookObject must have been created - see example in pymptk_decompose.cpp
PyObject * book_short_info(BookObject* self);

static PyMethodDef book_methods[] = {
	{"read", (PyCFunction)book_read, METH_VARARGS,
		"read a book file"
		},
	{"short_info", (PyCFunction)book_short_info, METH_NOARGS,
		"print short info"
		},
	{NULL}  /* Sentinel */
};


////////////////////////////////////////////////////
// module methods

PyObject * mptk_loadconfig(PyObject *self, PyObject *args);

PyObject * mptk_decompose(PyObject *self, PyObject *args, PyObject *keywds);

struct mptk_decompose_result { BookObject* thebook; PyArrayObject* residual; };
int mptk_decompose_body(const PyArrayObject *numpysignal, const char *dictpath, const int samplerate, const unsigned long int numiters, const float snr, const char *method, const bool getdecay, const char* bookpath, mptk_decompose_result& result);

PyObject * mptk_reconstruct(PyObject *self, PyObject *args);

MPTK_LIB_EXPORT extern PyArrayObject* mp_create_numpyarray_from_signal(MP_Signal_c *signal);
MPTK_LIB_EXPORT extern MP_Signal_c*   mp_create_signal_from_numpyarray(const PyArrayObject *nparray);

MPTK_LIB_EXPORT extern int mpbook_from_pybook(MP_Book_c *mpbook, BookObject* pybook);
MPTK_LIB_EXPORT extern MP_Atom_c *mpatom_from_pyatom(PyDictObject* pyatom, MP_Chan_t numChans, unsigned long int atomIdx);

#endif

