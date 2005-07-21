function [book] = bookread( fileName )

% BOOKREAD Imports a binary Matching Pursuit book in Matlab
%
%    book = BOOKREAD( 'fileName' ) reads the binary format book
%    file 'fileName' and returns it as a structure.
%
%    See also BOOKPLOT, BOOKOVER.

%%
%% Authors:
%% Sacha Krstulovic & R�mi Gribonval
%% Copyright (C) 2005 IRISA                                              
%%
%% This script is part of the Matching Pursuit Library package,
%% distributed under the General Public License.
%%
%% CVS log:
%%   $Author$
%%   $Date$
%%   $Revision$
%%

fid = fopen( fileName );
if (fid == -1),
   error( [ 'Can''t open file [' fileName ']' ] );
end;

% Get the format line
l = fgets(fid);
if strcmp(l,'bin\n'),
   error( 'Bad book file, or book is not in bin format.' );
end;

% Get the header
l = fgets( fid );
[a,c,e,nextIndex] = sscanf( l, '<book nAtom="%lu" numChans="%d" numSamples="%lu" sampleRate="%d" ');
if ( c < 4 ),
   fclose(fid);
   error('Failed to scan the header.');
end;
book.numAtoms   = a(1);
book.numChans   = a(2);
book.numSamples = a(3);
book.sampleRate = a(4);
[book.libVersion,c] = sscanf( l(nextIndex:end), 'libVersion="%[0-9a-z.]">\n' );
if ( c ~= 1 ),
   fclose(fid);
   error('Failed to scan the lib version.');
end;

% Get the atoms
for ( i = 1:book.numAtoms );

    % Get the atom type
    l = fgets( fid );
    if ( l == -1 ),
       fclose(fid);
       error( [ 'Can''t read an atom type for atom number [' num2str(i) '].' ] );
    end;
    atomType = sscanf( l, '%[a-z]\n' );
    book.atom{i}.type = atomType;

    % Get the generic atom parameters
    numChans = fread( fid, 1, 'int' );
    for ( c = 1:numChans ),
	pos(c) = fread( fid, 1, 'ulong' );
	len(c) = fread( fid, 1, 'ulong' );
    end;
    book.atom{i}.pos = pos;
    book.atom{i}.len = len;

    switch atomType,

	   case 'gabor',
	   l = fgets( fid );
	   book.atom{i}.windowType = sscanf( l, '%[a-z]\n' );
	   book.atom{i}.windowOpt = fread( fid, 1, 'double' );
	   book.atom{i}.freq  = fread( fid, 1, 'double' );
	   book.atom{i}.chirp = fread( fid, 1, 'double' );
	   book.atom{i}.amp   = fread( fid, numChans, 'double' );
	   book.atom{i}.phase = fread( fid, numChans, 'double' );

	   case 'harmonic',
	   l = fgets( fid );
	   book.atom{i}.windowType = sscanf( l, '%[a-z]\n' );
	   book.atom{i}.windowOpt = fread( fid, 1, 'double' );
	   book.atom{i}.freq  = fread( fid, 1, 'double' );
	   book.atom{i}.chirp = fread( fid, 1, 'double' );
	   book.atom{i}.amp   = fread( fid, numChans, 'double' );
	   book.atom{i}.phase = fread( fid, numChans, 'double' );
	   numPartials = fread( fid, 1, 'unsigned int' );
	   book.atom{i}.numPartials = numPartials;
	   book.atom{i}.harmonicity = fread( fid, numPartials, 'double' );
	   book.atom{i}.partialAmpStorage = fread( fid, numChans*numPartials, 'double' );
	   book.atom{i}.partialPhaseStorage = fread( fid, numChans*numPartials, 'double' );

	   case 'dirac',
	   book.atom{i}.amp   = fread( fid, numChans, 'double' );

	   % Unknown atom type
	   otherwise,
	   error( [ '[' atomType '] is an unknown atom type.'] );
    end;

end;

% Get the closing tag
l = fgets( fid );
if strcmp(l,'</book>\n'),
   warning( 'Failed to read the closing </book> tag.' );
end;

fclose(fid);
