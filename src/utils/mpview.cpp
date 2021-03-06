/******************************************************************************/
/*                                                                            */
/*                                mpview.cpp                                  */
/*                                                                            */
/*                        Matching Pursuit Utilities                          */
/*                                                                            */
/* RÈmi Gribonval                                                             */
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
/*
 * SVN log:
 *
 * $Author: broy $
 * $Date: 2007-07-13 16:24:32 +0200 (Fri, 13 Jul 2007) $
 * $Revision: 1112 $
 *
 */

#include <mptk.h>
#include <stdio.h>
#include <string.h>
#include "libgetopt/getopt.h"


const char* func = "mpview";

/********************/
/* Global constants */
/********************/
#define MPVIEW_TRUE  (1==1)
#define MPVIEW_FALSE (0==1)

/********************/
/* Error types      */
/********************/
#define ERR_ARG        1
#define ERR_BOOK       2
#define ERR_COL        3
#define ERR_ROW        4
#define ERR_PIX        5
#define ERR_BUILD      6
#define ERR_WRITE      7
#define ERR_LOADENV    8

/********************/
/* Global variables */
/********************/
int MPVIEW_QUIET      = MPVIEW_FALSE;
int MPVIEW_VERBOSE    = MPVIEW_FALSE;
int MPVIEW_WVMODE     = MPVIEW_FALSE;

/* Input/output file names: */
char			*bookFileName = NULL;
int				numCols = 640;
int				numRows = 480;
char			*pixFileName  = NULL;
const char		*configFileName = NULL;


/**************************************************/
/* HELP FUNCTION                                  */
/**************************************************/
void usage( void ) {
	fprintf( stdout, " \n" );
	fprintf( stdout, " Usage:\n" );
	fprintf( stdout, "     mpview [options] (bookFILE.bin|-) tfmapFILE.flt\n" );
	fprintf( stdout, " \n" );
	fprintf( stdout, " Synopsis:\n" );
	fprintf( stdout, "     Makes a time-frequency pixmap fill it with the time-frequency representation\n");
	fprintf( stdout, "     of the atoms contained in the book file bookFile.bin and write it to the file\n");
	fprintf( stdout, "     tfmapFILE.flt as a raw sequence of floats.\n");
	fprintf( stdout, "     The pixmap size is %dx%d pixels unless option --size is used.\n", numCols, numRows );
	fprintf( stdout, "     Each time-frequency atom is displayed with a rectangle unless\n");
	fprintf( stdout, "     unless option --wigner is used to display its pseudo Wigner-Ville distribution\n" );
	fprintf( stdout, " \n" );
	fprintf( stdout, " Mandatory arguments:\n" );
	fprintf( stdout, "     (bookFILE.bin|-)     A book of atoms, or stdin.\n" );
	fprintf( stdout, "     tfmapFILE.flt        The file where to write the pixmap in float.\n" );
	fprintf( stdout, " \n" );
	fprintf( stdout, " Optional arguments:\n" );
	fprintf( stdout, "     -C<FILE>, --config-file=<FILE>   Use the specified configuration file, otherwise MPTK_CONFIG_FILENAME\n" );
	fprintf( stdout, "     -s, --size=<numCols>x<numRows>   Change the size of the pixmap.\n" );
	fprintf( stdout, "     -w, --wigner						Change the display mode.\n" );
	fprintf( stdout, "     -q, --quiet						No text output.\n" );
	fprintf( stdout, "     -v, --verbose					Verbose.\n" );
	fprintf( stdout, "     -V, --version					Output the version and exit.\n" );
	fprintf( stdout, "     -h, --help						This help.\n" );
	fprintf( stdout, " \n" );
	exit(0);
}

/**************************************************/
/* PARSING OF THE ARGUMENTS                       */
/**************************************************/
int parse_args(int argc, char **argv) 
{
	int c, i;
	int val;
	char *p;
	char *ep;
	struct option longopts[] = 
	{
		{"config-file",  required_argument, NULL, 'C'},
		{"size",   required_argument, NULL, 's'},
		{"quiet",   no_argument, NULL, 'q'},
		{"wigner",   no_argument, NULL, 'w'},
		{"verbose", no_argument, NULL, 'v'},
		{"version", no_argument, NULL, 'V'},
		{"help",    no_argument, NULL, 'h'},
		{0, 0, 0, 0}
	};

	opterr = 0;
	optopt = '!';

	while ((c = getopt_long(argc, argv, "C:s:qwvVh", longopts, &i)) != -1 ) 
	{
		switch (c) 
		{
			case 'C':
				mp_debug_msg( MP_DEBUG_PARSE_ARGS, func, "switch -C : optarg is [%s].\n", optarg );
				if (optarg == NULL)
				{
					mp_error_msg( func, "After switch -C or switch --config-file=.\n" );
					mp_error_msg( func, "the argument is NULL.\n" );
					mp_error_msg( func, "(Did you use --config-file without the '=' character ?).\n" );
					return( ERR_ARG );
				}
				else 
					configFileName = optarg;
				mp_debug_msg( MP_DEBUG_PARSE_ARGS, func, "Read config-file name [%s].\n", configFileName );
				break;
			case 'h':
				usage();
				break;
			case 'q':
				MPVIEW_QUIET = MPVIEW_TRUE;
				mp_debug_msg( MP_DEBUG_PARSE_ARGS, func, "MPVIEW_QUIET is TRUE.\n" );
				break;
			case 'w':
				MPVIEW_WVMODE = MPVIEW_TRUE;
				mp_debug_msg( MP_DEBUG_PARSE_ARGS, func, "MPVIEW_WVMODE is TRUE.\n" );
				break;
			case 'v':
				MPVIEW_VERBOSE = MPVIEW_TRUE;
				mp_debug_msg( MP_DEBUG_PARSE_ARGS, func, "MPVIEW_VERBOSE is TRUE.\n" );
				break;
			case 'V':
				fprintf(stdout, "mpview -- Matching Pursuit library version %s\n", VERSION);
				exit(0);
				break;
			case 's':
				p = optarg;
				/* Get the numCols value */
				val = strtol( p, &ep, 0 );
				if ( p == ep ) 
				{
					mp_error_msg( func, "Could not read a numCols value in the size (%s, pointing at %s).", optarg, p );
					return( 1 );
				}
				else 
					numCols = val;
				/* Check the middle 'x' */
				p = ep;
				if ( *p != 'x' ) 
				{
					mp_error_msg( func, "Missing 'x' character between numCols and numRows in the size (%s, pointing at %s).", optarg, p );
					return( 1 );
				}
				/* Get the numRows value */
				p++; 
				val = strtol( p, &ep,0 );
				if ( p == ep ) 
				{
					mp_error_msg( func, "Could not read a numRows value in the size (%s, pointing at %s).", optarg, p );
					return( 1 );
				}
				else 
					numRows = val ;
				/* Check the ending ']' */
				p = ep;
				if (*p != '\0') 
				{
					mp_error_msg( func, "Spurious characters at the end of the size (%s, pointing at %s).", optarg, p );
					return( 1 );
				}
				break;
			default:
				mp_error_msg( func, "The command line contains the unrecognized option [%s].\n", argv[optind-1] );
				return( ERR_ARG );
		} /* end switch */
	} /* end while */

	mp_debug_msg( MP_DEBUG_PARSE_ARGS, func, "When exiting getopt, optind is [%d].\n", optind );
	mp_debug_msg( MP_DEBUG_PARSE_ARGS, func, "(argc is [%d].)\n", argc );

	/* Check if some file names / numbers are following the options */
	if ( (argc-optind) < 1 ) 
	{
		mp_error_msg( func, "You must indicate a file name (or - for stdin) for the book to use.\n");
		return( ERR_ARG );
	}
	if ( (argc-optind) < 2 ) 
	{
		mp_error_msg( func, "You must indicate a file name where to write the pixmap.\n");
		return( ERR_ARG );
	}

	/* Read the file names and numbers after the options */
	bookFileName = argv[optind++];
	mp_debug_msg( MP_DEBUG_PARSE_ARGS, func, "Read book file name [%s].\n", bookFileName );
	pixFileName = argv[optind++];
	mp_debug_msg( MP_DEBUG_PARSE_ARGS, func, "Read pixmap file name [%s].\n", pixFileName );

	/***********************/
	/* Basic options check */
	/* Can't have quiet AND verbose (make up your mind, dude !) */
	if ( MPVIEW_QUIET && MPVIEW_VERBOSE ) 
	{
		mp_error_msg( func, "Choose either one of --quiet or --verbose.\n");
		return( ERR_ARG );
	}
	return(0);
}


/**************************************************/
/* MAIN                                           */
/**************************************************/
int main( int argc, char **argv ) 
{
	MP_Book_c *book;
	int i;
   
	/* Parse the command line */
	if ( argc == 1 ) 
		usage();

	if ( parse_args( argc, argv ) ) 
	{
		mp_error_msg( func, "Please check the syntax of your command line. (Use --help to get some help.)\n" );
		fflush( stderr );
		exit( ERR_ARG );
	}

	/* Report */
	if ( !MPVIEW_QUIET ) 
	{
		mp_info_msg( func, "-------------------------------------\n" );
		mp_info_msg( func, "MPVIEW - MATCHING PURSUIT DISPLAY    \n" );
		mp_info_msg( func, "-------------------------------------\n" );
		mp_info_msg( func, "The command line was:\n" );
		for ( i = 0; i < argc; i++ ) 
			fprintf( stderr, "%s ", argv[i] );
		fprintf( stderr, "\n");
		mp_info_msg( func, "End command line.\n" );
		mp_info_msg( func, "Displaying book [%s] to pixmap [%s].\n", bookFileName, pixFileName );
	}
  
	/* Load the MPTK environment */
	if(! (MPTK_Env_c::get_env()->load_environment_if_needed(configFileName)) ) 
	{
		exit(ERR_LOADENV);
	}
  
	/* Make the book */
	book = MP_Book_c::create();
	if ( book == NULL ) 
	{
		mp_error_msg( func, "Can't create a new book.\n" );
		fflush( stderr );
		return( ERR_BOOK );
	}

	/* Read the book */
	if ( !strcmp( bookFileName, "-" ) ) 
	{
		if ( MPVIEW_VERBOSE ) 
			mp_info_msg( func, "Reading the book from stdin..." );
		if ( book->load(stdin) == 0 ) 
		{
			mp_error_msg( func, "No atoms were found in stdin.\n" );
			fflush( stderr );
			return( ERR_BOOK );
		}
	}
	else 
	{
		if ( MPVIEW_VERBOSE ) 
			mp_info_msg( func, "Reading the book from file [%s]...\n", bookFileName );
		if ( book->load( bookFileName ) == 0 ) 
		{
			mp_error_msg( func, "No atoms were found in the book file [%s].\n", bookFileName );
			fflush( stderr );
			return( ERR_BOOK );
		}
	}
	if ( MPVIEW_VERBOSE ) 
	{ 
		mp_info_msg( func, "Done.\n" ); 
		fflush( stderr ); 
	}

	/* Fill the pixmap */
	if ( MPVIEW_VERBOSE ) 
		mp_info_msg( func, "Initializing the pixmap..." );
	MP_TF_Map_c* tfmap = new MP_TF_Map_c( numCols, numRows, book->numChans, 0, book->numSamples, 0.0, 0.5 );
	if ( MPVIEW_VERBOSE ) 
		mp_info_msg( func, "Done.\n" );
	if ( MPVIEW_VERBOSE ) 
		mp_info_msg( func, "Filling the pixmap..." );

	char tfMapType = MP_TFMAP_SUPPORTS;
	if (MPVIEW_WVMODE) 
		tfMapType = MP_TFMAP_PSEUDO_WIGNER;
	if ( book->add_to_tfmap( tfmap, tfMapType, NULL ) == 0 ) 
	{
		mp_error_msg( func, "No atoms were found in the book to fill the pixmap.\n" );
		fflush( stderr );
		return( ERR_BUILD );
	}
	if ( MPVIEW_VERBOSE ) 
		mp_info_msg( func, "Done.\n" );
	/* Save the pixmap */
	if ( MPVIEW_VERBOSE ) 
		mp_info_msg( func, "Dumping the pixmap to file [%s]...\n", pixFileName );
	if ( tfmap->dump_to_file( pixFileName, 1 ) == 0 ) 
	{
		mp_error_msg( func, "Can't write filled pixmap to file [%s].\n", pixFileName );
		fflush( stderr );
		return( ERR_WRITE );
	}
	if ( MPVIEW_VERBOSE ) 
		mp_info_msg( func, "Done.\n" );
  
	/* End report */
	if ( !MPVIEW_QUIET ) 
	{
		mp_info_msg( func, "The resulting pixmap has size [%d x %d] pixels and [%d] channel(s).\n", numCols,numRows, (int)(book->numChans) );
		mp_info_msg( func, "Have a nice day !\n" );
	}
  
	fflush( stderr );
	fflush( stdout );

	/* Clean the house */
	delete( book );
	delete( tfmap );
	MPTK_Env_c::get_env()->release_environment();

	return(0);
}
