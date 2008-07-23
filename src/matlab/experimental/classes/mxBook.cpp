/******************************************************************************/
/*                                                                            */
/*                  	    mxBook.cpp                                        */
/*                                                                            */
/*          		    matlab4mptk toolbox									  */
/*                                                                            */
/*          Class for interfacing MP_Book with matlab strcture                */
/*                                                                            */
/* Gilles Gonon                                               	  Feb 20 2008 */
/* Remi Gribonval                                              	  July 2008   */
/* -------------------------------------------------------------------------- */
/*                                                                            */
/*  This program is free software; you can redistribute it and/or             */
/*  modify it under the terms of the GNU General Public License               */
/*  as published by the Free Software Foundation; either version 2            */
/*  of the License, or (at your option) any later version.                    */
/*                                                                            */
/*  This program is distributed in the hope that it will be useful,           */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/*  GNU General Public License for more details.                              */
/*                                                                            */
/*  You should have received a copy of the GNU General Public License         */
/*  along with this program; if not, write to the Free Software               */
/*  Foundation, Inc., 59 Temple Place - Suite 330,                            */
/*  Boston, MA  02111-1307, USA.                                              */
/*                                                                            */
/******************************************************************************/
/*
 * $Version 0.5.4$
 * $Date 05/22/2007$
 */

#include "mxBook.h"  


/***************************/
/*                         */
/*  CLASS mxAtoms METHODS  */
/*                         */
/***************************/

/** CONSTRUCTORS */
//! Default empty constructor
mxAtoms::mxAtoms() {} 

//! Constructor with no allocation of params mxArray
mxAtoms::mxAtoms(string t, unsigned int nC) : typeLen(t), typeIdx(0), nAtom(0), nChannel(nC), curIdx(0) 
{ 
  size_t i;
  if ((i = typeLen.rfind("_")) != string::npos) {  type = typeLen.substr(0,i); }
  else { type = typeLen; }
}

//! Constructor with correct allocation of params mxArray
mxAtoms::mxAtoms(string t,unsigned int tI,unsigned long int nA,unsigned int nC) : typeLen(t), typeIdx(tI), nAtom(nA), nChannel(nC), curIdx(0) 
{
  size_t i;
  if ((i = typeLen.rfind("_")) != string::npos) {  type = typeLen.substr(0,i); }
  else { type = typeLen; }

  
  // Init map for each parameters
  mxAtoms::allocParams(nAtom,nChannel);
}  /* end of constructor */

/** DESCTRUCTOR */
mxAtoms::~mxAtoms()
{
  /* todo */ 
}
  
/** OTHER MEMBER FUNCTIONS */

//! Allocate Atom matlab memory for each parameters
void mxAtoms::allocParams(unsigned long int nA,unsigned int nC) {
  // Create param container according to atom type
  // Default params Fieds
  //( OK for sType=="nyquist" || sType=="constant" || sType=="dirac" )
  params["amp"] = mxCreateDoubleMatrix(nA, nC, mxREAL);
  params["pos"] = mxCreateDoubleMatrix(nA, nC, mxREAL);
  params["len"] = mxCreateDoubleMatrix(nA, nC, mxREAL);
                
  // Add atom specific field to structure
  // Anywave atoms
  if (type=="anywave") {
    params["anywaveIdx"] = mxCreateDoubleMatrix(nA,1, mxREAL);   //! BAD VALUE TO FIX
    params["tableIdx"] = mxCreateDoubleMatrix(nA, 1, mxREAL);     //! BAD VALUE TO FIX
 //   params["anywaveTable"] = mxCreateDoubleMatrix(nA, 1, mxREAL); //! BAD VALUE TO FIX
  }
  // Anywave hilbert atoms
  else if (type=="anywavehilbert") {
    params["anywaveIdx"] = mxCreateDoubleMatrix(nA,1, mxREAL);   //! BAD VALUE TO FIX
    params["tableIdx"] = mxCreateDoubleMatrix(nA, 1, mxREAL);     //! BAD VALUE TO FIX
 //   params["anywaveTable"] = mxCreateDoubleMatrix(nA, 1, mxREAL); //! BAD VALUE TO FIX
    params["realTableIdx"] = mxCreateDoubleMatrix(nA, 1, mxREAL);       //! BAD VALUE TO FIX
 //   params["anywaveRealTable"] = mxCreateDoubleMatrix(nA, 1, mxREAL);   //! BAD VALUE TO FIX
    params["hilbertTableIdx"] = mxCreateDoubleMatrix(nA, 1, mxREAL);    //! BAD VALUE TO FIX
 //   params["anywaveHilbertTable"] = mxCreateDoubleMatrix(nA,1, mxREAL); //! BAD VALUE TO FIX
    params["realPart"] = mxCreateDoubleMatrix(nA, nC, mxREAL);           //! BAD VALUE TO FIX
    params["hilbertPart"] = mxCreateDoubleMatrix(nA, nC, mxREAL);        //! BAD VALUE TO FIX
  }
  // Atoms that use a analysis window
  else if (type=="mdct" || type=="mdst" || type=="gabor" || type=="mclt" || type=="harmonic") {
    params["freq"] = mxCreateDoubleMatrix(nA, 1, mxREAL);
    params["windowtype"] = mxCreateDoubleMatrix(nA, 1, mxREAL);
    params["windowoption"] = mxCreateDoubleMatrix(nA, 1, mxREAL);
    // Chirped atoms
    if (type=="gabor" || type=="mclt" || type=="harmonic") {
      params["chirp"] = mxCreateDoubleMatrix(nA, 1, mxREAL);
      params["phase"] = mxCreateDoubleMatrix(nA, nC, mxREAL);
      // Harmonic atoms
      if (type=="harmonic") {
	params["numPartials"] = mxCreateDoubleMatrix(nA, 1, mxREAL);
	// The following parameters can only be allocated when reading for the first time numPartials ... 
	//params["harmonicity"] = mxCreateDoubleMatrix(nA, nC, mxREAL);
	//params["partialAmp"] = mxCreateDoubleMatrix(nA, nC, mxREAL);
	//params["partialPhase"] = mxCreateDoubleMatrix(nA, nC, mxREAL);
      }
    }
  }
}
            
//! Read an atom and store values in params mxArrays
void mxAtoms::parseAtom(MP_Atom_c *atom) {
  unsigned int c,h,nP;
              char * func = "mxAtoms:parseAtom";  
  mp_debug_msg(func,"entering %s\n",type.c_str());
  /* CHANNEL INDEPENDENT PARAMETERS */
  /* ATOM Specific parameters */
  // Anywave atoms
  if (type=="anywave") {
    //MP_Anywave_Atom_Plugin_c * catom = (MP_Anywave_Atom_Plugin_c *) atom;
    *(mxGetPr( params["anywaveIdx"]) + curIdx ) = (double) atom->get_field(MP_ANYWAVE_IDX_PROP,0);
    *(mxGetPr( params["tableIdx"]) + curIdx ) = (double) atom->get_field(MP_TABLE_IDX_PROP,0);
  //  *(mxGetPr( params["anywaveTable"]) + curIdx ) = (double) 0.0; //! FAKE VALUE - NOT DONE
  }
  // Anywave hilbert atoms
  else if (type=="anywavehilbert") {
	//MP_Atom_c *catom = atom;
    MP_Anywave_Hilbert_Atom_Plugin_c * catom = (MP_Anywave_Hilbert_Atom_Plugin_c *) atom;
    *(mxGetPr( params["anywaveIdx"]) + curIdx ) = (double) catom->get_field(MP_ANYWAVE_IDX_PROP,0);
    *(mxGetPr( params["tableIdx"]) + curIdx ) = (double) catom->get_field(MP_TABLE_IDX_PROP,0);
  //  *(mxGetPr( params["anywaveTable"]) + curIdx ) = (double) 0.0; //! FAKE VALUE - NOT DONE
    *(mxGetPr( params["realTableIdx"]) + curIdx ) = (double) catom->get_field(MP_REAL_TABLE_IDX_PROP,0);
//    *(mxGetPr( params["anywaveRealTable"]) + curIdx ) = (double) 0.0;    //! FAKE VALUE - NOT DONE
    *(mxGetPr( params["hilbertTableIdx"]) + curIdx ) = (double) catom->get_field(MP_HILBERT_TABLE_IDX_PROP,0);
//    *(mxGetPr( params["anywaveHilbertTable"]) + curIdx ) = (double) 0.0; //! FAKE VALUE - NOT DONE
  }
  // Atoms that use a analysis window
  else if (type=="mdct" || type=="mdst" || type=="gabor" || type=="mclt" || type=="harmonic") {
    *(mxGetPr( params["freq"]) + curIdx ) = (double) atom->get_field(MP_FREQ_PROP,0);
    *(mxGetPr( params["windowtype"]) + curIdx ) = (double) atom->get_field(MP_WINDOW_TYPE_PROP,0);
    *(mxGetPr( params["windowoption"]) + curIdx ) = (double) atom->get_field(MP_WINDOW_OPTION_PROP,0);
    // Chirped atoms
    if (type=="gabor" || type=="mclt" || type=="harmonic") {
      *(mxGetPr( params["chirp"]) + curIdx ) = (double) atom->get_field(MP_CHIRP_PROP,0);
      // Harmonic atoms
      if (type=="harmonic") {
	nP = (unsigned int) atom->get_field(MP_NUMPARTIALS_PROP,0);
	*(mxGetPr( params["numPartials"]) + curIdx ) = (double) nP;
	/* Allocate params field with numPartials if necessary */
	if (params.find("harmonicity") == params.end()) {
	  params["harmonicity"] = mxCreateDoubleMatrix(nAtom, nP, mxREAL);
	  mwSize dims[3] = {nAtom,nP,nChannel};
	  params["partialAmp"] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);   //! channel dependent
	  params["partialPhase"] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL); //! channel dependent
	}
	// Read harmonicity values
	for (h=0;h<nP;h++) {
	  *( mxGetPr(params["harmonicity"]) + h*nAtom + curIdx ) = (double) atom->get_field(MP_HARMONICITY_PROP,h);
	}
      }
    }
  }
                                        
  /* CHANNEL DEPENDENT PARAMETERS */
  for (c=0;c<nChannel;c++) {
    /* ATOM Common parameters */
    *(mxGetPr( params["amp"]) + c*nAtom + curIdx ) = (double) atom->amp[c];
    *(mxGetPr( params["pos"]) + c*nAtom + curIdx ) = (double) atom->support[c].pos;
    *(mxGetPr( params["len"]) + c*nAtom + curIdx ) = (double) atom->support[c].len;
    /* ATOM Specific parameters */
    if (type=="anywavehilbert") {
 	//MP_Atom_c *catom = atom;
    MP_Anywave_Hilbert_Atom_Plugin_c * catom = (MP_Anywave_Hilbert_Atom_Plugin_c *) atom;
     *(mxGetPr( params["realPart"]) + c*nAtom + curIdx ) = (double) catom->get_field(MP_REAL_PART_PROP,c);
      *(mxGetPr( params["hilbertPart"]) + c*nAtom + curIdx ) = (double) catom->get_field(MP_HILBERT_PART_PROP,c);
    }
    // Atoms that have phase information
    // WAS // if (type=="mdct" || type=="mdst" || type=="gabor" || type=="mclt" || type=="harmonic") {
    else if (type=="gabor" || type=="mclt" || type=="harmonic") {
      *(mxGetPr( params["phase"]) + c*nAtom + curIdx ) = (double) atom->get_field(MP_PHASE_PROP,c);
      // Harmonic atoms
      if (type=="harmonic") {
	// Read Partial amp/phase info
	for (h=0;h<nP;h++) {
	  /* !! CHECK THESE VALUES */    *(mxGetPr( params["partialAmp"]) + c*(nAtom*nP) + h*nAtom + curIdx ) = (double) atom->get_field(MP_PARTIAL_AMP_PROP,h*nChannel+c);
	  /* !! CHECK THESE VALUES */    *(mxGetPr( params["partialPhase"])+ c*(nAtom*nP) + h*nAtom + curIdx ) = (double) atom->get_field(MP_PARTIAL_PHASE_PROP,h*nChannel+c);
	}
      }
    }
  }          
  curIdx++; //! Go to next index
}
            

//! Fill a given 'atom' structure at index 'a' with parameters
mxArray * mxAtoms::outputMxStruct(mxArray * atom, unsigned int a) {
  char * func = "mxAtoms::outputMxStruct";
  mp_debug_msg(func," - Fill atom Structure for type [%s]\n",typeLen.c_str()); 
  // Create atom Structure              
  mwSize dims[2] = {1, 2};
  mxArray *par;
  unsigned int p;
                
  mxSetField(atom, a, "type", mxCreateString(type.c_str()) );
                
  // Create parameters Structure (added to atom)
  dims[1] = 1; // change dimension [1x1] struct
  const char *emptyFieldNames[] = {""};
  par = mxCreateStructArray(2,dims,0,emptyFieldNames);  //! Parameter structure
                
  // Loop on map parameters: add field name and corresponding values to struct
  map <string, mxArray *>::iterator miter;
  for ( miter = params.begin(); miter != params.end(); miter++ )
    {
      mp_debug_msg(func,"   - \"%s\" added\n",miter->first.c_str());
      p = mxAddField(par,miter->first.c_str());
      mxSetFieldByNumber(par,0, p, miter->second);
    } // End of atom parameters definition
                
  mxSetField(atom, a, "params", par);
  return atom;
}


/**********************************/
/*                                */
/*  END OF CLASS mxAtoms METHODS  */
/*                                */
/**********************************/



/***************************/
/*                         */
/*  CLASS mxBook  METHODS  */
/*                         */
/***************************/

/** CONSTRUCTORS */
/** Constructor 1: copy mxArray pointer to book structure */
/* used by bookwrite.mex */
mxBook::mxBook(const mxArray * mxbook) {
  // Copy mex structure 
  mexbook = mxDuplicateArray(mxbook);
  
  mxArray *mxTmp;
  mxTmp = mxGetField(mexbook,0,"numAtoms");
  numAtoms = (unsigned long int)mxGetScalar(mxTmp);
  mxTmp = mxGetField(mexbook,0,"numChans");
  numChans = (MP_Chan_t)mxGetScalar(mxTmp);
  
}

/** Constructor 2: Construct mxArray book structure from MP_Book_c pointer */ 
/* used by bookread.mex */

mxBook::mxBook(MP_Book_c *mpbook) {
  
  char *func = "mxBook::mxBook(MP_Book_c)";
   numChans = mpbook->numChans;
  numAtoms = mpbook->numAtoms;
  
  // Allocate Output book structure
  int numBookFieldNames = 7;
  const char *bookFieldNames[] = {"format","numAtoms","numChans","numSamples","sampleRate","index","atom"};
  mexbook = mxCreateStructMatrix((mwSize)1,(mwSize)1,numBookFieldNames,bookFieldNames); //! Output book structure

  // Fill header info 
  mxArray *mxTmp;
  mxTmp = mxCreateString("0.1");
  mxSetField(mexbook,0, "format", mxTmp);
  mxTmp = mxCreateDoubleScalar((double) numAtoms); //! numAtoms
  mxSetField(mexbook, 0, "numAtoms", mxTmp);
  mxTmp = mxCreateDoubleScalar((double) numChans); //! numChans
  mxSetField(mexbook, 0, "numChans", mxTmp);
  mxTmp = mxCreateDoubleScalar((double) mpbook->numSamples); //! numSamples
  mxSetField(mexbook, 0, "numSamples", mxTmp);
  mxTmp = mxCreateDoubleScalar(mpbook->sampleRate); //! sampleRate
  mxSetField(mexbook, 0, "sampleRate", mxTmp);
    
  // Count atom types and fill index
  mxArray * mxIndex;
  unsigned int indexSize = 4+numChans;
  mxIndex = mxCreateDoubleMatrix((mwSize)indexSize,(mwSize)numAtoms, mxREAL); //! mxIndex contains (1: Atom number, 2: type index, 3: atom index, 4: Atom selected, 4+chan: atom pos of channel chan)
    
  mp_debug_msg(func,"Counting atom types in book : \n");
    
  // Some declarations
  map <string, mxAtoms *> atomStats;       //! Map <atom type , nb of occurence>
  map <string, mxAtoms *>::iterator miter; //! and a iterator on it
  mxArray *atom;
  unsigned long int n;

  for (n=0;n<numAtoms;n++) {
    // Get the aType string "<type_name>_<length>"
    string aType;
    stringstream ss1;
    ss1 << mpbook->atom[n]->type_name() << "_" <<  mpbook->atom[n]->support[0].len ;
    ss1 >> aType;

    // If aType is a new type, register it in maps
    if ( atomStats.find(aType) == atomStats.end() ) {
      mp_debug_msg(func,"Registering new atom type [%s]\n",aType.c_str());
      atomStats[aType] = new mxAtoms(aType,numChans);
    }
        
    atomStats[aType]->nAtom++; //! Increment atom count 
                
    //! Fill book index info for this atom
    *( mxGetPr(mxIndex) + n*indexSize ) = (double) (n+1);                         //! Atom number
    // As the map is not filled the typeIdx cannot be set at this stage
    // *( mxGetPr(mxIndex) + 4*n + 1 ) = (double) atomStats[aType]->typeIdx; //! Type index
    *( mxGetPr(mxIndex) + n*indexSize + 2 ) = (double) atomStats[aType]->nAtom;   //! Atom number
    // loop on channels to get their position
    *( mxGetPr(mxIndex) + n*indexSize + 3 ) = (double) 1.0;   //! Atom selected 
    // loop on channels to get their position
	unsigned short int c;
	for (c=0;c<numChans;c++) {
      *( mxGetPr(mxIndex) + n*indexSize + 4 + c) = (double) mpbook->atom[n]->support[c].pos;     //! Atom pos
    }
  }
    
 
  /* Matlab console info */
  mp_debug_msg(func, "found %d different atom types\n",atomStats.size());
    
  /* Init Atom structure
   *  atom.type = string
   *  atom.params = structure, size : 1*1 of atoms type found in book
   *       For each type of atom, params is a structure
   *       with the specified parameters (depend on atom type)
   */
    
    
  /** Allocate structure params for the different type of atoms
   *  and set the correct type index */
  unsigned int t = 0;  //! Index of type used for mxAtoms constructors
  for ( miter = atomStats.begin(); miter != atomStats.end(); ++miter )
    {
      mp_debug_msg(func," - atom [%s] :  %ld occurences\n",miter->first.c_str(), miter->second->nAtom);
      miter->second->allocParams(miter->second->nAtom, numChans);
      miter->second->typeIdx = t;  //! Set type index according to the map iterator
      t++; //! increment type index
    }

    
  mp_debug_msg(func,"Load each atom\n");

  /* Parse Atoms parameters */
  for ( n=0 ; n<numAtoms ; n++ ) {
    string aType;
    stringstream ss1;
    ss1 << mpbook->atom[n]->type_name() << "_" <<  mpbook->atom[n]->support[0].len ;
    ss1 >> aType;

    if ( atomStats.find(aType) != atomStats.end() ) {
      atomStats[aType]->parseAtom(mpbook->atom[n]);
      *( mxGetPr(mxIndex) + n*indexSize + 1 ) = (double) ( atomStats[aType]->typeIdx + 1); //! Type index
    } else {
      mp_error_msg(func,"Atom [%ld] was not recognized\n",n);
    }
  }
    
  // Add atom structure to output Variables
  int numAtomFieldNames = 2;
  const char *atomFieldNames[] = {"type","params"};
  atom = mxCreateStructMatrix((mwSize)1,(mwSize)(atomStats.size()),numAtomFieldNames,atomFieldNames);
    
  mp_debug_msg(func,"Creating output structure\n");
  n = 0;
  for ( miter = atomStats.begin(); miter != atomStats.end(); ++miter )
    {
      mp_debug_msg(func," - atom [%s] \n",miter->second->type.c_str());
      miter->second->outputMxStruct(atom,n);
      n++;
    } // End of atom parameters definition
    

  /** Add index info to book structure */ 
  mxSetField(mexbook, 0, "index", mxIndex);

  // Attach the atom structure to mexbook.atom
  mxSetField(mexbook, 0, "atom", atom);
}

/** DESTRUCTOR */
mxBook::~mxBook() {
  // todo
}


/** OTHER METHODS */


/** Get MP_Atom from mx book structure */
/** This maybe far from fast method (get atom by type) */
MP_Atom_c * mxBook::getMP_Atom(unsigned long int atomIdx) {
  char *func = "mxBook::getMP_Atom";
  // Add index info to book structure
  mxArray * mxIdx;
  unsigned int indexSize = 4+numChans;

  mxIdx = mxGetField(mexbook, 0, "index"); //! Matrix of size 4*nAtoms
  unsigned int t =  (unsigned int) *( mxGetPr(mxIdx) + atomIdx*indexSize + 1 ); //! typeMap index
  unsigned long int n =  (unsigned long int) *( mxGetPr(mxIdx) + atomIdx*indexSize + 2 ); //! atom number for given type
  MP_Chan_t c;
  
  // Get Atom type structure
  mxArray *mxAtom, *mxType, *mxParams;
  mxAtom = mxGetField(mexbook,0,"atom");      //! Structure with field 'type' 'params'
  mxParams = mxGetField(mxAtom,t-1,"params"); //! Structure with fields different for each type
  mxType = mxGetField(mxAtom,t-1,"type");     //! Get string type
  string aType(mxArrayToString(mxType));

  // Get Atom creator method 
  MP_Atom_c* (*emptyAtomCreator)( void ) = MP_Atom_Factory_c::get_atom_factory()->get_empty_atom_creator(aType.c_str());
  if (NULL == emptyAtomCreator)  {
      mp_error_msg(func,"-- unknown  MP_Atom_Factory_c method for atomType:%s\n",aType.c_str());
      return( NULL );
    }
  
  // Create empty atom 
  MP_Atom_c *newAtom = (*emptyAtomCreator)();
  if ( NULL==newAtom ) {
	mp_error_msg(func,"-- could not create empty atom of type %s\n",aType.c_str());
	return( NULL );
  }
  // Allocate main fields
  if ( newAtom->alloc_atom_param(numChans) ) {
	  mp_error_msg(func,"Failed to allocate some vectors in the atom %s. Returning a NULL atom.\n",aType.c_str());
	  delete newAtom;
	  return( NULL );
	}
  mp_debug_msg(func," -- atom index %ld [%s]\n",atomIdx,aType.c_str());
  
  /** Retrieve pointer for common parameters to all atoms (pos, len, amp) */
  mxArray *mxpos, *mxamp, *mxlen;
  mxpos = mxGetField(mxParams,0,"pos");
  mxlen = mxGetField(mxParams,0,"len");
  mxamp = mxGetField(mxParams,0,"amp");
   
  // number of atoms for this type (used for browsing mxArray)
  unsigned long int nA = (unsigned long int) mxGetM(mxamp); 

  // Fill the main fields
   for (c=0;c<numChans;c++) { // loop on channels
	newAtom->support[c].pos = (unsigned long int) *(mxGetPr(mxpos) + c*nA + (n-1));
	newAtom->support[c].len = (unsigned long int) *(mxGetPr(mxlen) + c*nA + (n-1));
	newAtom->totalChanLen += newAtom->support[c].len;
 	newAtom->amp[c] = (MP_Real_t)  *(mxGetPr(mxamp) + c*nA + (n-1));
	}
	
 
  // and fill it with the params field
  /** CONSTANT / DIRAC / NYQUIST /  */
  if (aType=="constant" || aType=="dirac" || aType=="nyquist") {
     return (newAtom);
    } 
  /** GABOR / HARMONIC */
  else if (aType=="gabor" || aType=="harmonic") {
	MP_Gabor_Atom_Plugin_c* gaborAtom =  (MP_Gabor_Atom_Plugin_c*)newAtom;
	if ( gaborAtom->alloc_gabor_atom_param(numChans) )	{
		mp_error_msg(func,"Failed to allocate some vectors in the atom %s. Returning a NULL atom.\n",aType.c_str());
		delete newAtom;
		return( NULL );
	}

	mxArray *mxchirp, *mxfreq, *mxphase, *mxwintype, *mxwinopt;
    mxchirp = mxGetField(mxParams,0,"chirp");
    mxfreq = mxGetField(mxParams,0,"freq");
	mxphase = mxGetField(mxParams,0,"phase");
	mxwintype = mxGetField(mxParams,0,"windowtype");
	mxwinopt = mxGetField(mxParams,0,"windowoption");
      
    gaborAtom->windowType = (unsigned char) *(mxGetPr(mxwintype) + (n-1));
	gaborAtom->windowOption = (double) *(mxGetPr(mxwinopt) + (n-1));
	gaborAtom->freq  = (MP_Real_t) *(mxGetPr(mxfreq) + (n-1));
	gaborAtom->chirp = (MP_Real_t) *(mxGetPr(mxchirp) + (n-1));
	for (c=0;c<numChans;c++) { // loop on channels
		gaborAtom->phase[c] = (MP_Real_t)  *(mxGetPr(mxphase) + c*nA + (n-1));
	}
      
	if(aType=="gabor") {
		return (newAtom);
	}
	else if (aType=="harmonic")  {
		MP_Harmonic_Atom_Plugin_c* harmonicAtom =  (MP_Harmonic_Atom_Plugin_c*)newAtom;

        mxArray *mxnumPartial;
		mxnumPartial = mxGetField(mxParams,0,"numPartials");
		unsigned int numPartials = (unsigned int) *(mxGetPr(mxnumPartial) + (n-1));
		unsigned int p;
		if ( harmonicAtom->alloc_harmonic_atom_param( numChans, numPartials ) ) {
			mp_error_msg(func,"Failed to allocate some vectors in the atom %s. Returning a NULL atom.\n",aType.c_str());
			delete newAtom;
			return( NULL );
		}
		/** set atoms params */
		mxArray *mxharmo, *mxpartialamp, *mxpartialphase;      
		mxharmo = mxGetField(mxParams,0,"harmonicity");
		mxpartialamp = mxGetField(mxParams,0,"partialAmp");
		mxpartialphase = mxGetField(mxParams,0,"partialPhase");
      
		harmonicAtom->numPartials = numPartials;
		for (p=0;p<numPartials;p++) { // loop on partials
			harmonicAtom->harmonicity[p] =  (MP_Real_t)  *(mxGetPr(mxharmo) + p*nA +  (n-1));	  
		}
		for (c=0;c<numChans;c++) { // loop on channels
			for (p=0;p<numPartials;p++) { // loop on partials
				harmonicAtom->partialAmp[c][p] = (MP_Real_t)  *(mxGetPr(mxpartialamp) + c*(nA*numPartials) + p*nA + (n-1)); // When reading was c*(nAtom*nP) + h*nAtom + curIdx
				harmonicAtom->partialPhase[c][p] = (MP_Real_t)  *(mxGetPr(mxpartialphase) + c*(nA*numPartials) + p*nA + (n-1));
			}
		}
		return (newAtom);
	} else {
		mp_error_msg(func,"This code should never be reached!\n");
		delete newAtom;
		return(NULL);
	}
  }
  /** ANYWAVE / ANYWAVE_HILBERT */
  else if (aType=="anywave" || aType=="anywavehilbert") {  
	MP_Anywave_Atom_Plugin_c* anywaveAtom =  (MP_Anywave_Atom_Plugin_c*)newAtom;
    mxArray *mxtableidx, *mxanywaveidx;
	mxanywaveidx = mxGetField(mxParams,0,"anywaveIdx");
	anywaveAtom->anywaveIdx =  (unsigned long int) *(mxGetPr(mxanywaveidx) + (n-1));
	mxtableidx = mxGetField(mxParams,0,"tableIdx");
	anywaveAtom->tableIdx =  (unsigned long int) *(mxGetPr(mxtableidx) + (n-1));
	anywaveAtom->anywaveTable = MPTK_Server_c::get_anywave_server()->tables[anywaveAtom->tableIdx];
	if(NULL==anywaveAtom->anywaveTable) {
		mp_error_msg(func,"Failed to retrieve anywaveTable number %d from server\n",anywaveAtom->tableIdx);
		delete newAtom;
		return(NULL);
	}
	if(aType=="anywave") {
      return (newAtom);
    } else if (aType=="anywavehilbert") {
		MP_Anywave_Hilbert_Atom_Plugin_c* anywaveHilbertAtom =  (MP_Anywave_Hilbert_Atom_Plugin_c*)newAtom;
		if ( anywaveHilbertAtom->alloc_hilbert_atom_param(numChans) ) {
			mp_error_msg(func,"Failed to allocate some vectors in the atom %s. Returning a NULL atom.\n",aType.c_str());
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
        anywaveHilbertAtom->realTableIdx =  (unsigned long int) *(mxGetPr(mxrealtableidx) + (n-1));
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
  
		anywaveHilbertAtom->hilbertTableIdx =  (unsigned long int) *(mxGetPr(mxhilberttableidx) + (n-1));
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
   /** MCLT ATOM */
  else if (aType=="mclt")  {
	MP_Mclt_Atom_Plugin_c* mcltAtom =  (MP_Mclt_Atom_Plugin_c*)newAtom;
	if ( mcltAtom->alloc_mclt_atom_param(numChans) ) {
		mp_error_msg(func,"Failed to allocate some vectors in the atom %s. Returning a NULL atom.\n",aType.c_str());
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
	mcltAtom->freq  = (MP_Real_t) *(mxGetPr(mxfreq) + (n-1));
	mcltAtom->chirp = (MP_Real_t) *(mxGetPr(mxchirp) + (n-1));
	for (c=0;c<numChans;c++) { // loop on channels
		mcltAtom->phase[c] = (MP_Real_t)  *(mxGetPr(mxphase) + c*nA + (n-1));
	}
      
      return (newAtom);
  }
  /** MDCT ATOM */
  else if (aType=="mdct") {
	MP_Mdct_Atom_Plugin_c* mdctAtom =  (MP_Mdct_Atom_Plugin_c*)newAtom;
	mxArray *mxfreq, *mxwintype, *mxwinopt;
	mxfreq = mxGetField(mxParams,0,"freq");
	mxwintype = mxGetField(mxParams,0,"windowtype");
	mxwinopt = mxGetField(mxParams,0,"windowoption");
      
	mdctAtom->windowType = (unsigned char) *(mxGetPr(mxwintype) + (n-1));
	mdctAtom->windowOption = (double) *(mxGetPr(mxwinopt) + (n-1));
	mdctAtom->freq  = (MP_Real_t) *(mxGetPr(mxfreq) + (n-1));
	return(newAtom);
  }
  /** MDST ATOM */
  else if (aType=="mdst") {
	MP_Mdst_Atom_Plugin_c* mdstAtom =  (MP_Mdst_Atom_Plugin_c*)newAtom;
	mxArray *mxfreq, *mxwintype, *mxwinopt;
	mxfreq = mxGetField(mxParams,0,"freq");
	mxwintype = mxGetField(mxParams,0,"windowtype");
	mxwinopt = mxGetField(mxParams,0,"windowoption");
      
	mdstAtom->windowType = (unsigned char) *(mxGetPr(mxwintype) + (n-1));
	mdstAtom->windowOption = (double) *(mxGetPr(mxwinopt) + (n-1));
	mdstAtom->freq  = (MP_Real_t) *(mxGetPr(mxfreq) + (n-1));
	return(newAtom);
  }
  else {
	mp_error_msg(func,"Atom type [%s] unknown, consider adding its information in mxBook{h,cpp}\n",aType.c_str());
    return (NULL);
  } 
}


/** Export matlab book structure to MP_Book_c class */
MP_Book_c * mxBook::Book_MEX_2_MPTK() {
    char * func = "mxBook::Book_MEX_2_MPTK";
  MP_Book_c * mpBook;
  mxArray *mxTmp,*atoms,*mxIndex;
  MP_Chan_t numChans;
  int sampleRate;
  unsigned long int nAtom,numSamples,a,nAtomAdded;
  unsigned int indexSize;
    
  /* Read matlab book structure header */
  mxTmp = mxGetField(mexbook,0,"numAtoms");
  nAtom = (unsigned long int)mxGetScalar(mxTmp);
  mxTmp = mxGetField(mexbook,0,"numChans");
  numChans = (MP_Chan_t)mxGetScalar(mxTmp);
  mxTmp = mxGetField(mexbook,0,"numSamples");
  numSamples = (unsigned long int)mxGetScalar(mxTmp);
  mxTmp = mxGetField(mexbook,0,"sampleRate");
  sampleRate = (int)mxGetScalar(mxTmp);

  /** Get book index */
  mxIndex = mxGetField(mexbook,0,"index");
  indexSize = 4+numChans;

  /** Create a empty book */
  mpBook = MP_Book_c::create(numChans,numSamples,sampleRate);
    
  // Add selected atoms to book
  nAtomAdded = 0;
  for (a=0;a<nAtom;a++) {
    if (*(mxGetPr(mxIndex) + a*indexSize + 3) != 0.0 ) {
      //! add_atom
      mp_debug_msg(func," - Adding Atom [%ld] to book :",a);
      MP_Atom_c * mpAtom;      
      if ( (mpAtom = this->getMP_Atom(a)) == NULL ) {
	mp_error_msg(func," getMP_Atom returend NULL while adding Atom [%ld] to book :",a);
      } else {
	mpBook->append( mpAtom );
	nAtomAdded++;
      }
    }
  }
    
  mp_info_msg(func," - [%ld] atoms have been added to book.\n",nAtomAdded);

  return mpBook;
}

mxArray *mp_create_mxBook_from_book(MP_Book_c *book) {
  // Load book object in Matlab structure
  mxBook * mexBook = new mxBook(book);
  if(NULL!= mexBook) {
    mxArray *res = mxDuplicateArray(mexBook->mexbook);
    return(res);
  }
  else {
    return(NULL);
  }
}

MP_Book_c *mp_create_book_from_mxBook(const mxArray *mexBook)  {
  // Load book structure in object 
  mxBook mybook(mexBook);
  MP_Book_c* book =  mybook.Book_MEX_2_MPTK();
  return(book);
}



