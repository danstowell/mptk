#include "pyMPTK.h"

void
book_dealloc(book* self)
{
	self->ob_type->tp_free((PyObject*)self);
}

PyObject *
book_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	/* Load the MPTK environment if not loaded */
  MPTK_Env_c::get_env()->load_environment_if_needed("");
	book *self;

	self = (book *)type->tp_alloc(type, 0);
	self->book = MP_Book_c::create();
	self->numAtoms = 0;
	self->numChans = 0;
	self->numSamples = 0;
	self->sampleRate = 44100;
	self->atomType = PyList_New(0);
	self->atomParams = PyDict_New();

	return (PyObject *)self;
}

int
book_init(book *self, PyObject *args, PyObject *kwds)
{
	return 0;
}

PyObject *
book_read(book* self, PyObject *args)
{
	char *filename;
	int n,m;
	PyObject *tmp,*len,*pos,*freq,*amp,*phase,*chirp;

	if (!PyArg_ParseTuple(args, "s", &filename))
		return NULL;

	self->book->load( filename );

	self->numAtoms = self->book->numAtoms;
	self->numChans = self->book->numChans;
	self->numSamples = self->book->numSamples;
	self->sampleRate = self->book->sampleRate;
	len = PyList_New(0);
	pos = PyList_New(0);
	freq = PyList_New(0);
	amp = PyList_New(0);
	phase = PyList_New(0);
	chirp = PyList_New(0);
	for ( n=0 ; n<self->numAtoms ; n++ ) {
		PyList_Append( self->atomType, Py_BuildValue("s", self->book->atom[n]->type_name()));
		tmp = PyList_New(0);
		for ( m=0 ; m<self->numChans ; m++ ) {
			if ( self->book->atom[n]->has_field(0) ) {
				PyList_Append( tmp, Py_BuildValue("i", (int)self->book->atom[n]->get_field(0,m)));
			}
		}
		PyList_Append( len, tmp);
		tmp = PyList_New(0);
		for ( m=0 ; m<self->numChans ; m++ ) {
			if ( self->book->atom[n]->has_field(1) ) {
				PyList_Append( tmp, Py_BuildValue("i", (int)self->book->atom[n]->get_field(1,m)));
			}
		}
		PyList_Append( pos, tmp);
		tmp = PyList_New(0);
		for ( m=0 ; m<self->numChans ; m++ ) {
			if ( self->book->atom[n]->has_field(2) ) {
				PyList_Append( tmp, Py_BuildValue("d", self->book->atom[n]->get_field(2,m)));
			}
		}
		PyList_Append( freq, tmp);
		tmp = PyList_New(0);
		for ( m=0 ; m<self->numChans ; m++ ) {
			if ( self->book->atom[n]->has_field(3) ) {
				PyList_Append( tmp, Py_BuildValue("d", self->book->atom[n]->get_field(3,m)));
			}
		}
		PyList_Append( amp, tmp);
		tmp = PyList_New(0);
		for ( m=0 ; m<self->numChans ; m++ ) {
			if ( self->book->atom[n]->has_field(4) ) {
				PyList_Append( tmp, Py_BuildValue("d", self->book->atom[n]->get_field(4,m)));
			}
		}
		PyList_Append( phase, tmp);
		tmp = PyList_New(0);
		for ( m=0 ; m<self->numChans ; m++ ) {
			if ( self->book->atom[n]->has_field(5) ) {
				PyList_Append( tmp, Py_BuildValue("d", self->book->atom[n]->get_field(5,m)));
			}
		}
		PyList_Append( chirp, tmp);
	}
	PyDict_SetItemString(self->atomParams, "len", len);
	PyDict_SetItemString(self->atomParams, "pos", pos);
	PyDict_SetItemString(self->atomParams, "freq", freq);
	PyDict_SetItemString(self->atomParams, "amp", amp);
	PyDict_SetItemString(self->atomParams, "phase", phase);
	PyDict_SetItemString(self->atomParams, "chirp", chirp);

	return Py_BuildValue("i", 0);
}

PyObject *
book_short_info(book* self)
{
	self->book->short_info();

	return Py_BuildValue("i", 0);
}

