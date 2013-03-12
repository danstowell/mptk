#include "pyMPTK.h"

void
book_dealloc(BookObject* pybook)
{
	pybook->ob_type->tp_free((PyObject*)pybook);
}

PyObject *
book_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	MPTK_Env_c::get_env()->load_environment_if_needed("");
	BookObject *pybook;

	pybook = (BookObject *)type->tp_alloc(type, 0);
	// NB the values below, pybook_from_mpbook() will overwrite them iff 0 so we should leave them here as 0
	pybook->mpbook = MP_Book_c::create(0, 0, 0);
	pybook->atoms = PyList_New(0);

	return (PyObject *)pybook;
}

int
book_init(BookObject *pybook, PyObject *args, PyObject *kwds)
{
	return 0;
}

PyObject *
book_read(BookObject* pybook, PyObject *args)
{
	char *filename;

	if (!PyArg_ParseTuple(args, "s", &filename))
		return NULL;

	pybook->mpbook->load( filename );

	int result = pybook_from_mpbook(pybook, pybook->mpbook);
	return Py_BuildValue("i", result);
}

int
pybook_from_mpbook(BookObject* pybook, MP_Book_c *mpbook)
{
	int numAtoms = mpbook->numAtoms;
	int n,m;
	PyObject *tmp;
	if(PyList_Size(pybook->atoms) != 0){
		printf("Attempted to load mpbook data into a pybook which is not empty.\n");
		return 4;
	}
	if(pybook->numChans==0){
		pybook->numChans = mpbook->numChans;
	}else if(pybook->numChans != mpbook->numChans ){
		return 1;
	}
	if(pybook->numSamples==0){
		pybook->numSamples = mpbook->numSamples;
	}else if(pybook->numSamples != mpbook->numSamples ){
		return 2;
	}
	if(pybook->sampleRate==0){
		pybook->sampleRate = mpbook->sampleRate;
	}else if(pybook->sampleRate != mpbook->sampleRate ){
		return 3;
	}
	for ( n=0 ; n<numAtoms ; ++n ) {

		// Create a dict representing one atom, containing all its properties
		PyObject* atom = PyDict_New();
		/////////////////////////////////
		// Mono properties:
		PyDict_SetItemString(atom, "type", Py_BuildValue("s", mpbook->atom[n]->type_name()));
		// freq
		if ( mpbook->atom[n]->has_field(MP_FREQ_PROP) ) {
			PyDict_SetItemString(atom, "freq", Py_BuildValue("d", mpbook->atom[n]->get_field(MP_FREQ_PROP, 0)));
		}
		// chirp
		if ( mpbook->atom[n]->has_field(MP_CHIRP_PROP) ) {
			PyDict_SetItemString(atom, "chirp", Py_BuildValue("d", mpbook->atom[n]->get_field(MP_CHIRP_PROP, 0)));
		}
		/////////////////////////////////
		// Multichannel properties:
		// len
		tmp = PyList_New(0);
		for ( m=0 ; m<pybook->numChans ; m++ ) {
			if ( mpbook->atom[n]->has_field(MP_LEN_PROP) ) {
				PyList_Append( tmp, Py_BuildValue("i", (int)mpbook->atom[n]->get_field(MP_LEN_PROP, m)));
			}
		}
		PyDict_SetItemString(atom, "len", tmp);
		// pos
		tmp = PyList_New(0);
		for ( m=0 ; m<pybook->numChans ; m++ ) {
			if ( mpbook->atom[n]->has_field(MP_POS_PROP) ) {
				PyList_Append( tmp, Py_BuildValue("i", (int)mpbook->atom[n]->get_field(MP_POS_PROP, m)));
			}
		}
		PyDict_SetItemString(atom, "pos", tmp);
		// amp
		tmp = PyList_New(0);
		for ( m=0 ; m<pybook->numChans ; m++ ) {
			if ( mpbook->atom[n]->has_field(MP_AMP_PROP) ) {
				PyList_Append( tmp, Py_BuildValue("d", mpbook->atom[n]->get_field(MP_AMP_PROP, m)));
			}
		}
		PyDict_SetItemString(atom, "amp", tmp);
		// phase
		tmp = PyList_New(0);
		for ( m=0 ; m<pybook->numChans ; m++ ) {
			if ( mpbook->atom[n]->has_field(MP_PHASE_PROP) ) {
				PyList_Append( tmp, Py_BuildValue("d", mpbook->atom[n]->get_field(MP_PHASE_PROP, m)));
			}
		}
		PyDict_SetItemString(atom, "phase", tmp);
		// and finally, append our atom to the list
		PyList_Append(pybook->atoms, atom);
	}
	return 0;
}

/* Writes a pyatom's data to an XML fragment in-memory */
void pyatom_innerxml(PyDictObject* atom, char* str, size_t maxlen){
	size_t writepos=0;
	
}

int
mpbook_from_pybook(MP_Book_c *mpbook, BookObject* pybook, MP_Dict_c* dict)
{
	// Given an mpbook already "create"d, this fills it in

	unsigned long int pynatoms = PyList_Size(pybook->atoms);
	if(pynatoms == 0){
		printf("Attempted to load mpbook data from a pybook which is empty.\n");
		return 4;
	}
	if(mpbook->numAtoms != 0){
		// TODO LATER: could use mpbook->reset()
		printf("Attempted to load data from a pybook into an mpbook which is not empty.\n");
		return 3;
	}
	if(mpbook->maxNumAtoms > pynatoms){
		printf("Attempted to load %i atoms from a pybook into an mpbook which can only hold %i atoms.\n", pynatoms, mpbook->maxNumAtoms);
		return 2;
	}


	mpbook->numChans   = pybook->numChans;
	mpbook->numSamples = pybook->numSamples;
	mpbook->sampleRate = pybook->sampleRate;

	for(unsigned long int i=0; i < pynatoms; ++i){
		// read atom from pybook, create one in mpbook
		PyObject* obj = PyList_GetItem(pybook->atoms, (Py_ssize_t)i);
		if(!PyDict_Check(obj)){
			printf("Error -- iterating atoms in book, found entry %i is not a dict\n", i);
			return 1;
		}
		PyDictObject* pyatom = (PyDictObject*)obj;

/*
		// Scan the hash map to get the create function of the atom
		MP_Atom_c* (*createAtom)( FILE *fid, MP_Dict_c *dict, const char mode ) = MP_Atom_Factory_c::get_atom_factory()->get_atom_creator( str );
		// Scan the hash map to get the create function of the atom
		if ( NULL != createAtom ) 
			// Create the the atom
			newAtom = (*createAtom)(fid,dict,mode);
		else 
			mp_error_msg( func, "Cannot read atoms of type '%s'\n",str);
	  
		if ( NULL == newAtom )  
			mp_error_msg( func, "Failed to create an atom of type[%s].\n", str);
*/

		
	}

	mpbook->recheck_num_samples();
	mpbook->recheck_num_channels();

	return 0;
}

PyObject *
book_short_info(BookObject* pybook)
{
	pybook->mpbook->short_info();
	return Py_BuildValue("i", 0);
}

