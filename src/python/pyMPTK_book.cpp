#include "pyMPTK.h"

void
book_dealloc(book* self)
{
	self->ob_type->tp_free((PyObject*)self);
}

PyObject *
book_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	MPTK_Env_c::get_env()->load_environment_if_needed("");
	book *self;

	self = (book *)type->tp_alloc(type, 0);
	self->book = MP_Book_c::create();
	self->numChans = 0;
	self->numSamples = 0;
	self->sampleRate = 44100;
	self->atoms = PyList_New(0);

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
	PyObject *tmp;

	if (!PyArg_ParseTuple(args, "s", &filename))
		return NULL;

	self->book->load( filename );

	int numAtoms = self->book->numAtoms;
	self->numChans = self->book->numChans;
	self->numSamples = self->book->numSamples;
	self->sampleRate = self->book->sampleRate;

	for ( n=0 ; n<numAtoms ; ++n ) {

		// Create a dict representing one atom, containing all its properties
		PyObject* atom = PyDict_New();
		PyDict_SetItemString(atom, "type", Py_BuildValue("s", self->book->atom[n]->type_name()));
		// len
		tmp = PyList_New(0);
		for ( m=0 ; m<self->numChans ; m++ ) {
			if ( self->book->atom[n]->has_field(0) ) {
				PyList_Append( tmp, Py_BuildValue("i", (int)self->book->atom[n]->get_field(0,m)));
			}
		}
		PyDict_SetItemString(atom, "len", tmp);
		// pos
		tmp = PyList_New(0);
		for ( m=0 ; m<self->numChans ; m++ ) {
			if ( self->book->atom[n]->has_field(1) ) {
				PyList_Append( tmp, Py_BuildValue("i", (int)self->book->atom[n]->get_field(1,m)));
			}
		}
		PyDict_SetItemString(atom, "pos", tmp);
		// freq
		tmp = PyList_New(0);
		for ( m=0 ; m<self->numChans ; m++ ) {
			if ( self->book->atom[n]->has_field(2) ) {
				PyList_Append( tmp, Py_BuildValue("d", self->book->atom[n]->get_field(2,m)));
			}
		}
		PyDict_SetItemString(atom, "freq", tmp);
		// amp
		tmp = PyList_New(0);
		for ( m=0 ; m<self->numChans ; m++ ) {
			if ( self->book->atom[n]->has_field(3) ) {
				PyList_Append( tmp, Py_BuildValue("d", self->book->atom[n]->get_field(3,m)));
			}
		}
		PyDict_SetItemString(atom, "amp", tmp);
		// phase
		tmp = PyList_New(0);
		for ( m=0 ; m<self->numChans ; m++ ) {
			if ( self->book->atom[n]->has_field(4) ) {
				PyList_Append( tmp, Py_BuildValue("d", self->book->atom[n]->get_field(4,m)));
			}
		}
		PyDict_SetItemString(atom, "phase", tmp);
		// chirp
		tmp = PyList_New(0);
		for ( m=0 ; m<self->numChans ; m++ ) {
			if ( self->book->atom[n]->has_field(5) ) {
				PyList_Append( tmp, Py_BuildValue("d", self->book->atom[n]->get_field(5,m)));
			}
		}
		PyDict_SetItemString(atom, "chirp", tmp);
		// and finally, append our atom to the list
		PyList_Append(self->atoms, atom);
	}

	return Py_BuildValue("i", 0);
}

PyObject *
book_short_info(book* self)
{
	self->book->short_info();

	return Py_BuildValue("i", 0);
}

