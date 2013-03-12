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

// Method to create an atom in mptk data structure, from a python specification in memory.
// Based on the matlab wrapper: MP_Atom_c *GetMP_Atom(const mxArray *mxBook,MP_Chan_t numChans,unsigned long int atomIdx)
MP_Atom_c* mpatom_from_pyatom(PyDictObject* pyatom, MP_Chan_t numChans, unsigned long int atomIdx) {
	const char *func = "mpatom_from_pyatom";
	PyObject* pyatomobj = (PyObject*)pyatom;
	unsigned long int c; //MP_Chan_t c;
/*	
	// Add index info to book structure
	mxArray * mxIdx;
	unsigned int indexSize = 4+numChans;

	mxIdx = mxGetField(mxBook, 0, "index"); //! Matrix of size 4*nAtoms
	unsigned int t =	(unsigned int) *( mxGetPr(mxIdx) + atomIdx*indexSize + 1 ); //! typeMap index
	unsigned long int n =	(unsigned long int) *( mxGetPr(mxIdx) + atomIdx*indexSize + 2 ); //! atom number for given type
	
	// Get Atom type structure
	mxArray *mxAtom, *mxType, *mxParams;
	mxAtom = mxGetField(mxBook,0,"atom");			//! Structure with field 'type' 'params'
	mxParams = mxGetField(mxAtom,t-1,"params"); //! Structure with fields different for each type
	mxType = mxGetField(mxAtom,t-1,"type");		 //! Get string type
	string aType(mxArrayToString(mxType));

*/

	PyObject* gotobj;
	PyObject* keyobj;
	char* keystr;

	keystr = "type";
	keyobj = PyString_FromString(keystr);
	gotobj = PyObject_GetItem(pyatomobj, keyobj);
	if(gotobj==NULL){
		printf("'type' gotobj is null\n");
		return NULL;
	}
	if(!PyString_Check(gotobj)){
		printf("'type' gotobj is not a string\n");
		return NULL;
	}
	const char* typestr = PyString_AsString(gotobj);


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
	keystr = "pos";
	keyobj = PyString_FromString(keystr);
	PyObject* listobj_pos = PyObject_GetItem(pyatomobj, keyobj);
	if(listobj_pos==NULL){
		printf("'pos' gotobj is null\n");
		return NULL;
	}
	if(!PyList_Check(listobj_pos)){
		printf("'pos' gotobj is not a list\n");
		return NULL;
	}
	//
	keystr = "len";
	keyobj = PyString_FromString(keystr);
	PyObject* listobj_len = PyObject_GetItem(pyatomobj, keyobj);
	if(listobj_len==NULL){
		printf("'len' gotobj is null\n");
		return NULL;
	}
	if(!PyList_Check(listobj_len)){
		printf("'len' gotobj is not a list\n");
		return NULL;
	}
	//
	keystr = "amp";
	keyobj = PyString_FromString(keystr);
	PyObject* listobj_amp = PyObject_GetItem(pyatomobj, keyobj);
	if(listobj_amp==NULL){
		printf("'amp' gotobj is null\n");
		return NULL;
	}
	if(!PyList_Check(listobj_amp)){
		printf("'amp' gotobj is not a list\n");
		return NULL;
	}

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

		keystr = "chirp";
		keyobj = PyString_FromString(keystr);
		PyObject* chirpobj = PyObject_GetItem(pyatomobj, keyobj);
		if(chirpobj==NULL){
			printf("'chirp' gotobj is null\n");
			return NULL;
		}
		if(!PyFloat_Check(chirpobj)){
			printf("'chirp' gotobj is not a float\n");
			return NULL;
		}
		//
		keystr = "freq";
		keyobj = PyString_FromString(keystr);
		PyObject* freqobj = PyObject_GetItem(pyatomobj, keyobj);
		if(freqobj==NULL){
			printf("'freq' gotobj is null\n");
			return NULL;
		}
		if(!PyFloat_Check(freqobj)){
			printf("'freq' gotobj is not a float\n");
			return NULL;
		}
		//
		keystr = "phase";
		keyobj = PyString_FromString(keystr);
		PyObject* phaseobj = PyObject_GetItem(pyatomobj, keyobj);
		if(phaseobj==NULL){
			printf("'phase' gotobj is null\n");
			return NULL;
		}
		if(!PyList_Check(phaseobj)){
			printf("'phase' gotobj is not a list\n");
			return NULL;
		}
		//
		keystr = "wintype";
		keyobj = PyString_FromString(keystr);
		PyObject* wintypeobj = PyObject_GetItem(pyatomobj, keyobj);
		if(wintypeobj==NULL){
			printf("'wintype' gotobj is null\n");
			return NULL;
		}
		if(!PyString_Check(wintypeobj)){
			printf("'wintype' gotobj is not a string\n");
			return NULL;
		}
		//

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
/* TODO
		else if (strcmp(typestr, "harmonic")==0)	{
			MP_Harmonic_Atom_Plugin_c* harmonicAtom =	(MP_Harmonic_Atom_Plugin_c*)newAtom;

					mxArray *mxnumPartial;
			mxnumPartial = mxGetField(mxParams,0,"numPartials");
			unsigned int numPartials = (unsigned int) *(mxGetPr(mxnumPartial) + (n-1));
			unsigned int p;
			if ( harmonicAtom->alloc_harmonic_atom_param( numChans, numPartials ) ) {
				mp_error_msg(func,"Failed to allocate some vectors in the atom %s. Returning a NULL atom.\n", typestr);
				delete newAtom;
				return( NULL );
			}
			// set atoms params 
			mxArray *mxharmo, *mxpartialamp, *mxpartialphase;			
			mxharmo = mxGetField(mxParams,0,"harmonicity");
			mxpartialamp = mxGetField(mxParams,0,"partialAmp");
			mxpartialphase = mxGetField(mxParams,0,"partialPhase");
			
			harmonicAtom->numPartials = numPartials;
			for (p=0;p<numPartials;p++) { // loop on partials
				harmonicAtom->harmonicity[p] =	(MP_Real_t)	*(mxGetPr(mxharmo) + p*nA +	(n-1));		
			}
			for (c=0;c<numChans;c++) { // loop on channels
				for (p=0;p<numPartials;p++) { // loop on partials
					harmonicAtom->partialAmp[c][p] = (MP_Real_t)	*(mxGetPr(mxpartialamp) + c*(nA*numPartials) + p*nA + (n-1)); // When reading was c*(nAtom*nP) + h*nAtom + curIdx
					harmonicAtom->partialPhase[c][p] = (MP_Real_t)	*(mxGetPr(mxpartialphase) + c*(nA*numPartials) + p*nA + (n-1));
				}
			}
			return (newAtom);
		}
*/
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

