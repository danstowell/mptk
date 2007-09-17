/******************************************************************************/
/*                                                                            */
/*                                 book.h                                     */
/*                                                                            */
/*                        Matching Pursuit Library                            */
/*                                                                            */
/* R�mi Gribonval                                                             */
/* Sacha Krstulovic                                           Mon Feb 21 2005 */
/* -------------------------------------------------------------------------- */
/*                                                                            */
/*  Copyright (C) 2005 IRISA                                                  */
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

/***************************************/
/*                                     */
/* DEFINITION OF THE BOOK CLASS        */
/*                                     */
/***************************************/
/*
 * SVN log:
 *
 * $Author$
 * $Date$
 * $Revision$
 *
 */


#ifndef __book_h_
#define __book_h_


/***********************/
/* CONSTANTS           */
/***********************/

/** \brief The minimum increase in the allocation size of a book. */
#define MP_BOOK_GRANULARITY 1024


/***********************/
/* BOOK CLASS          */
/***********************/
/**
 * \brief Books store and manage sequences of atoms collected
 * while iterating several steps of Matching Pursuit. 
 */
class MP_Book_c {

  /********/
  /* DATA */
  /********/

public:
  /** \brief Number of atoms actually stored in the book */
  unsigned long int numAtoms;
  /** \brief Number of atoms that could fit in the storage space */
  unsigned long int maxNumAtoms;
  /** \brief Storage space for atoms */
  MP_Atom_c **atom;

  /** \brief Number of channels of the atoms */
  MP_Chan_t numChans;                 
  /** \brief Number of samples in each channel of the signal that will be
   * reconstructed using the stored atoms */
  unsigned long int numSamples; 
  /** \brief Sample rate in Hertz of the analyzed signal */
  int sampleRate;


  /***********/
  /* METHODS */
  /***********/

  /***************************/
  /* CONSTRUCTORS/DESTRUCTOR */
  /***************************/

public:
  static MP_Book_c* create();
  static MP_Book_c* create(MP_Chan_t numChans, unsigned long int numSamples, int sampleRate );
  static MP_Book_c* create( FILE *fid );
private:
  /* NULL constructor */
  MP_Book_c();

public:
  /* Destructor */
  ~MP_Book_c();


  /***************************/
  /* I/O METHODS             */
  /***************************/

  /** \brief Print the book to a stream in text or binary format
   *
   * \param  fid A writable stream
   * \param  mode One of MP_TEXT or MP_BINARY
   * \param  mask a MP_Mask_c object telling which atoms should be used 
   * \return The number of atoms actually printed to the stream 
   * \remark Passing mask == NULL forces all atoms to be used. 
   */
  unsigned long int print( FILE *fid , const char mode, MP_Mask_c* mask);

  /** \brief Same as MP_Book_c::print (FILE *fid, const char mode, char* mask) but with a file name */
  unsigned long int print( const char *fName, const char mode, MP_Mask_c* mask);

  /** \brief Same as MP_Book_c::print( FILE *fid, const char mode, char *mask) with mask == NULL */
  unsigned long int print( FILE *fid, const char mode );

  /** \brief Same as MP_Book_c::print (FILE *fid, const char mode) but with a file name */
  unsigned long int print( const char *fName, const char mode );

  /** \brief Load the book from a stream, determining automatically
   * whether it is in text or binary format
   *
   * \param  fid A readable stream
   * \return The number of atoms loaded from the stream */
  unsigned long int load( FILE *fid );

  /** \brief Same as MP_Book_c::load (FILE *fid) but with a file name */
  unsigned long int load( const char *fName );

  /** \brief Print readable information about the book to a stream
   *
   * \param  fid A writable stream */
  int info( FILE *fid );

  /** \brief Print readable information about the book to the default info handler,
   *  including info about every atom in the book
   *
   */
  int info();

  /** \brief Print readable information about the book to the default info handler,
   *  in a short form (no atom info)
   *
   */
  int short_info();


  /***************************/
  /* MISC METHODS            */
  /***************************/

  /** \brief Clear all the atoms from the book.
   */
  void reset( void );

  /** \brief Add a new atom in the storage space, taking care of the necessary allocations 
   * \param newAtom a reference to an atom
   * \return the number of appended atoms (1 upon success, zero otherwise)
   * \remark The reference newAtom is not copied, it is stored and will be deleted when the book is destroyed
   * \remark \a numChans is set up if this is the first atom to be appended to the book,
   * otherwise, if the atom \a numChans does not match \a numChans, it is not appended.
   */
   
   int append( MP_Atom_c *newAtom );
   
   /** \brief Add a new atom in the storage space, taking care of the necessary allocations 
   * \param newAtom a reference to an atom
   * \return the number of appended atoms (1 upon success, zero otherwise)
   * \remark Use int append( MP_Atom_c *newAtom )
   */
  
  unsigned long int append( MP_Book_c *newBook );
  /** \brief Re-check the number of samples
   * \return MP_TRUE if numSamples was up to date, MP_FALSE if numSamples has been updated.
   */
  int recheck_num_samples();

  /** \brief Re-check the number of channels
   * \return MP_TRUE if numChans was up to date, MP_FALSE if numChans has been updated.
   */
  int recheck_num_channels();


  /** \brief Substract / add some atom waveforms from / to a signal 
   *
   * \param sigSub signal from which the sum of atom waveforms is to be removed
   * \param sigAdd signal to which the sum of atom waveforms is to be added
   * \param mask a MP_Mask_c object indicating which atoms should be used 
   * \return the number of atoms used
   * \remark Passing sigSub == NULL or sigAdd == NULL skips the corresponding substraction / addition.
   * \remark Passing mask == NULL forces all atoms to be used.
   * \remark An exception (assert) is throwed if the signals numChans does not match the atoms numChans.
   * \remark An exception (assert) is throwed if the support of the atom exceeds the limits of the signal.
   */
  unsigned long int substract_add( MP_Signal_c *sigSub, MP_Signal_c *sigAdd, MP_Mask_c *mask );

  /** \brief Build the waveform corresponding to the sum of some atoms of the book into a signal 
   *
   * \param sig signal to which the atom waveform is to be added
   * \param mask a MP_Mask_c object indicating which atoms should be used 
   * \return the number of atoms used
   * \remark The signal numChans, numSamples and sampleRate are set according to those of the book.
   * \remark Passing mask == NULL forces all atoms to be used.
   */
  unsigned long int build_waveform( MP_Signal_c *sig, MP_Mask_c *mask );

  /** \brief Adds the sum of the pseudo Wigner-Ville distributions of some atoms to a time-frequency map 
   * \param tfmap The time-frequency map 
   * \param mask a MP_Mask_c object indicating which atoms should be used
   * \param tfmapType an indicator of what to put in the tfmap, to be chosen among
   * MP_TFMAP_SUPPORTS or MP_TFMAP_PSEUDO_WIGNER (see tfmap.h for more).
   * \return the number of atoms used
   * \remark Passing mask == NULL forces all atoms to be used.
   */
  unsigned long int add_to_tfmap( MP_TF_Map_c *tfmap, const char tfmapType, MP_Mask_c *mask );

  /** \brief Returns the atom which is the closest to a given 
   *  time-frequency location, as well as its index in the book->atom[] array
   *  Masked atoms are not considered.
   * \param time the time-location, in sample coordinates
   * \param freq the frequency-location, between 0 and 0.5
   * \param chanIdx the considered channel
   * \param mask a mask indicating which atoms are considered (set it to NULL to consider all atoms)
   * \param n points to the index of the resulting atom in the book->atom[] array. Meaningless if no matching atom is found.
   * \return atom, the closest atom, NULL if no matching atom is found
   */
  MP_Atom_c *get_closest_atom(MP_Real_t time,MP_Real_t freq,
			      MP_Chan_t chanIdx, MP_Mask_c *mask,
			      unsigned long int *n);
			      
/** \brief Open a new book and check if this book have the same dimensions. */
  MP_Bool_t can_append( FILE * fid );

  /** \brief Check if numAtoms is the same in a mask and in the book. */
  MP_Bool_t is_compatible_with( MP_Mask_c mask );
  /** \brief Check between two books. */
  MP_Bool_t is_compatible_with( MP_Book_c *book );
  /** \brief Check if parameters are the same. */
  MP_Bool_t is_compatible_with(MP_Chan_t testedNumChans, int testedSampleRate);  
  /** \brief Check between a books and a signal. */
  MP_Bool_t is_compatible_with( MP_Signal_c *sig );  
};


#endif /* __book_h_ */