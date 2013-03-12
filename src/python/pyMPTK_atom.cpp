#include "pyMPTK.h"

#include "../plugin/base/anywave_atom_plugin.h"
#include "../plugin/base/anywave_hilbert_atom_plugin.h"
#include "../plugin/base/constant_atom_plugin.h"
#include "../plugin/base/dirac_atom_plugin.h"
#include "../plugin/base/gabor_atom_plugin.h"
#include "../plugin/base/harmonic_atom_plugin.h"
#include "../plugin/base/nyquist_atom_plugin.h"
#include "../plugin/contrib/lam/mclt_atom_plugin.h"
#include "../plugin/contrib/lam/mdct_atom_plugin.h"
#include "../plugin/contrib/lam/mdst_atom_plugin.h"

// macro used in mpatom_from_pyatom()
#define PYATOMOBJ_GETITEM(keystr, objvarname, checktype) \
		keyobj = PyString_FromString(keystr); \
		PyObject* objvarname = PyObject_GetItem(pyatomobj, keyobj); \
		Py_DECREF(keyobj); \
		if(objvarname==NULL){ \
			printf("'%s' gotobj is null\n", keystr); \
			return NULL; \
		} \
		if(!checktype##_Check(objvarname)){ \
			printf("'%s' gotobj is not of correct type\n", keystr); \
			return NULL; \
		}

// Method to create an atom in mptk data structure, from a python specification in memory.
// Based on the matlab wrapper: MP_Atom_c *GetMP_Atom(const mxArray *mxBook,MP_Chan_t numChans,unsigned long int atomIdx)
MP_Atom_c* mpatom_from_pyatom(PyDictObject* pyatom, MP_Chan_t numChans, unsigned long int atomIdx) {
	const char *func = "mpatom_from_pyatom";
	PyObject* pyatomobj = (PyObject*)pyatom;
	unsigned long int c; //MP_Chan_t c;

	PyObject* keyobj; // used in PYATOMOBJ_GETITEM
	char* keystr;

	PYATOMOBJ_GETITEM("type", typeobj, PyString)
	const char* typestr = PyString_AsString(typeobj);


	// Get Atom creator method 
	MP_Atom_c* (*emptyAtomCreator)( void ) = MP_Atom_Factory_c::get_atom_factory()->get_empty_atom_creator(typestr);
	if (NULL == emptyAtomCreator)	{
			printf("-- unknown	MP_Atom_Factory_c method for atomType:%s\n", typestr);
			return( NULL );
	}
	
	// Create empty atom 
	MP_Atom_c *newAtom = (*emptyAtomCreator)();
	if ( NULL==newAtom ) {
		mp_error_msg(func,"-- could not create empty atom of type %s\n", typestr);
		return( NULL );
	}
	// Allocate main fields
	if ( newAtom->alloc_atom_param(numChans) ) {
		mp_error_msg(func,"Failed to allocate some vectors in the atom %s. Returning a NULL atom.\n", typestr);
		delete newAtom;
		return( NULL );
	}
	mp_debug_msg(MP_DEBUG_SPARSE,func," -- atom index %ld [%s]\n",atomIdx, typestr);

	// retrieve pointers to arrays of parameters that are common to every atom type (pos, len, amp)
	PYATOMOBJ_GETITEM("pos", listobj_pos, PyList)
	PYATOMOBJ_GETITEM("len", listobj_len, PyList)
	PYATOMOBJ_GETITEM("amp", listobj_amp, PyList)

	// Fill the main fields
	for (c=0; c<numChans; ++c) { // loop on channels
		newAtom->support[c].pos = (unsigned long int) PyInt_AsLong(PyList_GetItem(listobj_pos, c));
		newAtom->support[c].len = (unsigned long int) PyInt_AsLong(PyList_GetItem(listobj_len, c));
		newAtom->totalChanLen += newAtom->support[c].len;
		newAtom->amp[c]         = (MP_Real_t)     PyFloat_AsDouble(PyList_GetItem(listobj_amp, c));
	}
	
	// and fill it with the params field
	// CONSTANT / DIRAC / NYQUIST / 
	if (strcmp(typestr, "constant")==0 || strcmp(typestr, "dirac")==0 || strcmp(typestr, "nyquist")==0) {
		 return (newAtom);
	} 

	// GABOR / HARMONIC
	else if (strcmp(typestr, "gabor")==0 || strcmp(typestr, "harmonic")==0) {
		MP_Gabor_Atom_Plugin_c* gaborAtom =	(MP_Gabor_Atom_Plugin_c*)newAtom;
		if ( gaborAtom->alloc_gabor_atom_param(numChans) )	{
			mp_error_msg(func,"Failed to allocate some vectors in the atom %s. Returning a NULL atom.\n", typestr);
			delete newAtom;
			return( NULL );
		}

		PYATOMOBJ_GETITEM("chirp",   chirpobj,   PyFloat)
		PYATOMOBJ_GETITEM("freq",    freqobj,    PyFloat)
		PYATOMOBJ_GETITEM("phase",   phaseobj,   PyList )
		PYATOMOBJ_GETITEM("wintype", wintypeobj, PyString)

		const char* wintypestr = PyString_AsString(wintypeobj);
		gaborAtom->windowType = window_type(wintypestr);

		gaborAtom->windowOption = 0.; // TODO
		gaborAtom->freq	= (MP_Real_t) PyFloat_AsDouble(freqobj);
		gaborAtom->chirp = (MP_Real_t) PyFloat_AsDouble(chirpobj);
		for (c=0;c<numChans;c++) { // loop on channels
			gaborAtom->phase[c] = (MP_Real_t) PyFloat_AsDouble(PyList_GetItem(phaseobj, c));
		}
			
		if(strcmp(typestr, "gabor")==0) {
			return (newAtom);
		}
		else if (strcmp(typestr, "harmonic")==0)	{
			MP_Harmonic_Atom_Plugin_c* harmonicAtom =	(MP_Harmonic_Atom_Plugin_c*)newAtom;


			PYATOMOBJ_GETITEM("numPartials",   numpartialsobj,   PyInt)

			unsigned int numPartials = (unsigned int) PyInt_AsLong(numpartialsobj);
			unsigned int p;
			if ( harmonicAtom->alloc_harmonic_atom_param( numChans, numPartials ) ) {
				mp_error_msg(func,"Failed to allocate some vectors in the atom %s. Returning a NULL atom.\n", typestr);
				delete newAtom;
				return( NULL );
			}


			// set atoms params 
			PYATOMOBJ_GETITEM("harmonicity",  harmonicityobj,  PyList)
			PYATOMOBJ_GETITEM("partialAmp",   partialampobj,   PyList)
			PYATOMOBJ_GETITEM("partialPhase", partialphaseobj, PyList)

			harmonicAtom->numPartials = numPartials;
			
			for (p=0;p<numPartials;p++) { // loop on partials
				harmonicAtom->harmonicity[p] =	(MP_Real_t) PyFloat_AsDouble(PyList_GetItem(harmonicityobj, p));
			}
			for (c=0;c<numChans;c++) { // loop on channels
				for (p=0;p<numPartials;p++) { // loop on partials
					harmonicAtom->partialAmp[c][p] = (MP_Real_t)   PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(partialampobj, c), p));
					harmonicAtom->partialPhase[c][p] = (MP_Real_t) PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(partialphaseobj, c), p));
				}
			}
			return (newAtom);
		}
		else {
			mp_error_msg(func,"This code should never be reached!\n");
			delete newAtom;
			return(NULL);
		}
	}
/* TODO these types should be easy to convert
	// ANYWAVE / ANYWAVE_HILBERT
	else if (strcmp(typestr, "anywave")==0 || strcmp(typestr, "anywavehilbert")==0) {	
		MP_Anywave_Atom_Plugin_c* anywaveAtom =	(MP_Anywave_Atom_Plugin_c*)newAtom;
			mxArray *mxtableidx, *mxanywaveidx;
		mxanywaveidx = mxGetField(mxParams,0,"anywaveIdx");
		anywaveAtom->anywaveIdx =	(unsigned long int) *(mxGetPr(mxanywaveidx) + (n-1));
		mxtableidx = mxGetField(mxParams,0,"tableIdx");
		anywaveAtom->tableIdx =	(unsigned long int) *(mxGetPr(mxtableidx) + (n-1));
		anywaveAtom->anywaveTable = MPTK_Server_c::get_anywave_server()->tables[anywaveAtom->tableIdx];
		if(NULL==anywaveAtom->anywaveTable) {
			mp_error_msg(func,"Failed to retrieve anywaveTable number %d from server\n",anywaveAtom->tableIdx);
			delete newAtom;
			return(NULL);
		}
		if(strcmp(typestr, "anywave")==0) {
				return (newAtom);
			} else if (strcmp(typestr, "anywavehilbert")==0) {
			MP_Anywave_Hilbert_Atom_Plugin_c* anywaveHilbertAtom =	(MP_Anywave_Hilbert_Atom_Plugin_c*)newAtom;
			if ( anywaveHilbertAtom->alloc_hilbert_atom_param(numChans) ) {
				mp_error_msg(func,"Failed to allocate some vectors in the atom %s. Returning a NULL atom.\n", typestr);
				delete newAtom;
				return( NULL );
			}			
			// Channel independent fields
			mxArray *mxrealtableidx = mxGetField(mxParams,0,"realTableIdx");
			if(NULL== mxrealtableidx) {
				mp_error_msg(func,"Could not retrieve realTableIdx\n");
				delete newAtom;
				return(NULL);
			}
					anywaveHilbertAtom->realTableIdx =	(unsigned long int) *(mxGetPr(mxrealtableidx) + (n-1));
			anywaveHilbertAtom->anywaveRealTable = MPTK_Server_c::get_anywave_server()->tables[anywaveHilbertAtom->realTableIdx];
			if(NULL==anywaveHilbertAtom->anywaveRealTable) {
				mp_error_msg(func,"Failed to retrieve anywaveRealTable number %d from server\n",anywaveHilbertAtom->realTableIdx);
				delete newAtom;
				return(NULL);
			}
	 
			mxArray *mxhilberttableidx = mxGetField(mxParams,0,"hilbertTableIdx");
				if(NULL== mxhilberttableidx) {
				mp_error_msg(func,"Could not retrieve hilbertTableIdx\n");
				delete newAtom;
				return(NULL);
			}
	
			anywaveHilbertAtom->hilbertTableIdx =	(unsigned long int) *(mxGetPr(mxhilberttableidx) + (n-1));
			anywaveHilbertAtom->anywaveHilbertTable = MPTK_Server_c::get_anywave_server()->tables[anywaveHilbertAtom->hilbertTableIdx];
			if(NULL==anywaveHilbertAtom->anywaveHilbertTable) {
				mp_error_msg(func,"Failed to retrieve anywaveHilbertTable number %d from server\n",anywaveHilbertAtom->hilbertTableIdx);
				delete newAtom;
				return(NULL);
			}
	 
			// Channel dependent fields
			mxArray *mxrealpart = mxGetField(mxParams,0,"realPart");
			if(NULL== mxrealpart) {
				mp_error_msg(func,"Could not retrieve realPart\n");
				delete newAtom;
				return(NULL);
			}
			mxArray *mxhilbertpart = mxGetField(mxParams,0,"hilbertPart");
			if(NULL== mxhilbertpart) {
				mp_error_msg(func,"Could not retrieve hilbertPart\n");
				delete newAtom;
				return(NULL);
			}
			for (c=0;c<numChans;c++) { // loop on channels
				// Real/Hilbert part
				anywaveHilbertAtom->realPart[c] = (MP_Real_t) *(mxGetPr(mxrealpart) + c*nA + (n-1));
				anywaveHilbertAtom->hilbertPart[c] = (MP_Real_t) *(mxGetPr(mxhilbertpart) + c*nA + (n-1));
			}
		 
			return(newAtom);
		} else {
			mp_error_msg(func,"This code should never be reached!\n");
			delete newAtom;
			return(NULL);
		}
	}
	 // MCLT ATOM
	else if (strcmp(typestr, "mclt")==0)	{
		MP_Mclt_Atom_Plugin_c* mcltAtom =	(MP_Mclt_Atom_Plugin_c*)newAtom;
		if ( mcltAtom->alloc_mclt_atom_param(numChans) ) {
			mp_error_msg(func,"Failed to allocate some vectors in the atom %s. Returning a NULL atom.\n", typestr);
			return( NULL );
		}

		mxArray *mxchirp, *mxfreq, *mxphase, *mxwintype, *mxwinopt;
		mxchirp = mxGetField(mxParams,0,"chirp");
		mxfreq = mxGetField(mxParams,0,"freq");
		mxphase = mxGetField(mxParams,0,"phase");
		mxwintype = mxGetField(mxParams,0,"windowtype");
		mxwinopt = mxGetField(mxParams,0,"windowoption");
			
		mcltAtom->windowType = (unsigned char) *(mxGetPr(mxwintype) + (n-1));
		mcltAtom->windowOption = (double) *(mxGetPr(mxwinopt) + (n-1));
		mcltAtom->freq	= (MP_Real_t) *(mxGetPr(mxfreq) + (n-1));
		mcltAtom->chirp = (MP_Real_t) *(mxGetPr(mxchirp) + (n-1));
		for (c=0;c<numChans;c++) { // loop on channels
			mcltAtom->phase[c] = (MP_Real_t)	*(mxGetPr(mxphase) + c*nA + (n-1));
		}
		return (newAtom);
	}
	// MDCT ATOM
	else if (strcmp(typestr, "mdct")==0) {
		MP_Mdct_Atom_Plugin_c* mdctAtom =	(MP_Mdct_Atom_Plugin_c*)newAtom;
		mxArray *mxfreq, *mxwintype, *mxwinopt;
		mxfreq = mxGetField(mxParams,0,"freq");
		mxwintype = mxGetField(mxParams,0,"windowtype");
		mxwinopt = mxGetField(mxParams,0,"windowoption");
			
		mdctAtom->windowType = (unsigned char) *(mxGetPr(mxwintype) + (n-1));
		mdctAtom->windowOption = (double) *(mxGetPr(mxwinopt) + (n-1));
		mdctAtom->freq	= (MP_Real_t) *(mxGetPr(mxfreq) + (n-1));
		return(newAtom);
	}
	// MDST ATOM
	else if (strcmp(typestr, "mdst")==0) {
		MP_Mdst_Atom_Plugin_c* mdstAtom =	(MP_Mdst_Atom_Plugin_c*)newAtom;
		mxArray *mxfreq, *mxwintype, *mxwinopt;
		mxfreq = mxGetField(mxParams,0,"freq");
		mxwintype = mxGetField(mxParams,0,"windowtype");
		mxwinopt = mxGetField(mxParams,0,"windowoption");
			
		mdstAtom->windowType = (unsigned char) *(mxGetPr(mxwintype) + (n-1));
		mdstAtom->windowOption = (double) *(mxGetPr(mxwinopt) + (n-1));
		mdstAtom->freq	= (MP_Real_t) *(mxGetPr(mxfreq) + (n-1));
		return(newAtom);
	}
*/
	else {
		mp_error_msg(func,"Atom type [%s] unknown, consider adding its information in pyMPTK_atom.cpp\n", typestr);
		return (NULL);
	} 
}

