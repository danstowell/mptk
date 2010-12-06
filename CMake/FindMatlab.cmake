# - Try to find a version of Matlab and headers/library required by the 
#   used compiler. It determines the right MEX-File extension and add 
#   a macro to help the build of MEX-functions.
#
# This module detects a Matlab's version between Matlab 7.0
# and Matlab 7.9
#
# This module defines: 
#  MATLAB_ROOT: Matlab installation path
#  MATLAB_INCLUDE_DIR: include path for mex.h, engine.h
#  MATLAB_MEX_LIBRARY: path to libmex.lib
#  MATLAB_MX_LIBRARY:  path to libmx.lib
#  MATLAB_ENG_LIBRARY: path to libeng.lib
#  MATLAB_LIBRARIES:   required libraries: libmex, libmx, libeng
#  MEX_EXTENSION: MEX extension required for the current plateform
#  MATLAB_CREATE_MEX: macro to build a MEX-file
#
# The macro MATLAB_CREATE_MEX requires in this order:
#  - function's name which will be called in Matlab;
#  - C/C++ source files;
#  - third libraries required.

IF(MATLAB_ROOT AND MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)
	# in cache already
	SET(Matlab_FIND_QUIETLY TRUE)
ENDIF(MATLAB_ROOT AND MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)

SET(LIBMEX "mex")
SET(LIBMX "mx")
SET(LIBENG "eng")

IF(PIPOL_IMAGE)
	IF(WIN32)
		IF(PIPOL_IMAGE MATCHES "amd64")
			SET(MATLAB_MEX_COMPILER "Y:/amd64/matlab-2010a-windows/bin/mex.bat")
			SET(MATLAB_MEX_LIBRARY "Y:/amd64/matlab-2010a-windows/extern/lib/win64/microsoft/libmex.lib")
			SET(MATLAB_MX_LIBRARY "Y:/amd64/matlab-2010a-windows/extern/lib/win64/microsoft/libmx.lib")
			SET(MATLAB_ENG_LIBRARY "Y:/amd64/matlab-2010a-windows/extern/lib/win64/microsoft/libeng.lib")
			SET(MATLAB_INCLUDE_DIR "Y:/amd64/matlab-2010a-windows/extern/include")
			SET(MEX_EXTENSION mexw64)
		ELSE(PIPOL_IMAGE MATCHES "amd64")
			SET(MATLAB_MEX_COMPILER "Y:/i386/matlab-2010a-windows/bin/mex.bat")
			SET(MATLAB_MEX_LIBRARY "Y:/i386/matlab-2010a-windows/extern/lib/win32/microsoft/libmex.lib")
			SET(MATLAB_MX_LIBRARY "Y:/i386/matlab-2010a-windows/extern/lib/win32/microsoft/libmx.lib")
			SET(MATLAB_ENG_LIBRARY "Y:/i386/matlab-2010a-windows/extern/lib/win32/microsoft/libeng.lib")
			SET(MATLAB_INCLUDE_DIR "Y:/i386/matlab-2010a-windows/extern/include")
			SET(MEX_EXTENSION mexw32)
		ENDIF(PIPOL_IMAGE MATCHES "amd64")
	ENDIF(WIN32)
	IF(UNIX)
		IF(APPLE)
			IF(CMAKE_SIZEOF_VOID_P MATCHES "4")
				SET(MATLAB_MEX_COMPILER "/netshare/x86_mac/matlab-2010a-mac/MATLAB_R2010a.app/bin/mex")
				SET(MATLAB_MEX_LIBRARY "/netshare/x86_mac/matlab-2010a-mac/MATLAB_R2010a.app/bin/maci/libmex.dylib")
				SET(MATLAB_MX_LIBRARY "/netshare/x86_mac/matlab-2010a-mac/MATLAB_R2010a.app/bin/maci/libmx.dylib")
				SET(MATLAB_ENG_LIBRARY "/netshare/x86_mac/matlab-2010a-mac/MATLAB_R2010a.app/bin/maci/libeng.dylib")
				SET(MATLAB_INCLUDE_DIR "/netshare/x86_mac/matlab-2010a-mac/MATLAB_R2010a.app/extern/include")
				SET(MEX_EXTENSION mexmaci)
			ELSE(CMAKE_SIZEOF_VOID_P MATCHES "4")
				SET(MATLAB_MEX_COMPILER "/netshare/x86_64_mac/matlab-2010a-mac/MATLAB_R2010a.app/bin/mex")
				SET(MATLAB_MEX_LIBRARY "/netshare/x86_64_mac/matlab-2010a-mac/MATLAB_R2010a.app/bin/maci64/libmex.dylib")
				SET(MATLAB_MX_LIBRARY "/netshare/x86_64_mac/matlab-2010a-mac/MATLAB_R2010a.app/bin/maci64/libmx.dylib")
				SET(MATLAB_ENG_LIBRARY "/netshare/x86_64_mac/matlab-2010a-mac/MATLAB_R2010a.app/bin/maci64/libeng.dylib")
				SET(MATLAB_INCLUDE_DIR "/netshare/x86_64_mac/matlab-2010a-mac/MATLAB_R2010a.app/extern/include")
				SET(MEX_EXTENSION mexmaci64)
			ENDIF(CMAKE_SIZEOF_VOID_P MATCHES "4")
		ELSE(APPLE)
			IF(CMAKE_SIZEOF_VOID_P MATCHES "4")
				SET(MATLAB_MEX_COMPILER "/netshare/i386/matlab-2010a-linux/bin/mex")
				SET(MATLAB_MEX_LIBRARY "/netshare/i386/matlab-2010a-linux/bin/glnx86/libmex.so")
				SET(MATLAB_MX_LIBRARY "/netshare/i386/matlab-2010a-linux/bin/glnx86/libmx.so")
				SET(MATLAB_ENG_LIBRARY "/netshare/i386/matlab-2010a-linux/bin/glnx86/libeng.so")
				SET(MATLAB_INCLUDE_DIR "/netshare/i386/matlab-2010a-linux/extern/include")
				SET(MEX_EXTENSION mexglx)
			ELSE(CMAKE_SIZEOF_VOID_P MATCHES "4")
				SET(MATLAB_MEX_COMPILER "/netshare/amd64/matlab-2010a-linux/bin/mex")
				SET(MATLAB_MEX_LIBRARY "/netshare/amd64/matlab-2010a-linux/bin/glnxa64/libmex.so")
				SET(MATLAB_MX_LIBRARY "/netshare/amd64/matlab-2010a-linux/bin/glnxa64/libmx.so")
				SET(MATLAB_ENG_LIBRARY "/netshare/amd64/matlab-2010a-linux/bin/glnxa64/libeng.so")
				SET(MATLAB_INCLUDE_DIR "/netshare/amd64/matlab-2010a-linux/extern/include")
				SET(MEX_EXTENSION mexa64)
			ENDIF(CMAKE_SIZEOF_VOID_P MATCHES "4")
		ENDIF(APPLE)
	ENDIF(UNIX)

ELSE(PIPOL_IMAGE)
	IF(UNIX)
		IF(APPLE)
			FILE(GLOB MATLAB_PATHS "/Applications/MATLAB_*")
			FOREACH(COUNT ${MATLAB_PATHS})
				IF("${COUNT}" STRGREATER "${MATLAB_PATH}")
					SET(MATLAB_PATH "${COUNT}")
				ENDIF("${COUNT}" STRGREATER "${MATLAB_PATH}")
			ENDFOREACH(COUNT ${MATLAB_PATHS})
			FIND_PATH(MATLAB_ROOT "license.txt" ${MATLAB_PATH})
			IF(NOT MATLAB_ROOT)
				MESSAGE("Unix users have to set the Matlab installation path into the MATLAB_ROOT variable.")
			ENDIF(NOT MATLAB_ROOT)
			IF(CMAKE_SIZEOF_VOID_P MATCHES "4")
	  			# 32 bits OS
				IF(CMAKE_SYSTEM_PROCESSOR MATCHES ppc)
					SET(MATLAB_LIBRARIES_PATHS "${MATLAB_ROOT}/bin/mac")
					SET(MEX_EXTENSION mexmac)
				ELSE(CMAKE_SYSTEM_PROCESSOR MATCHES ppc)
					SET(MATLAB_LIBRARIES_PATHS "${MATLAB_ROOT}/bin/maci")
					SET(MEX_EXTENSION mexmaci)
				ENDIF(CMAKE_SYSTEM_PROCESSOR MATCHES ppc)
			ELSE(CMAKE_SIZEOF_VOID_P MATCHES "4")
	  			# 64 bits OS
				IF(EXISTS "${MATLAB_ROOT}/bin/maci64/")
					SET(MATLAB_LIBRARIES_PATHS "${MATLAB_ROOT}/bin/maci64")
					SET(MEX_EXTENSION mexmaci64)
				ELSE(EXISTS "${MATLAB_ROOT}/bin/maci64/")
					MESSAGE("You use a Matlab 32 bits version on a 64 bits OS. MPTK will be compiled in 32 bits to be useble with your Matlab version.")
					SET(MATLAB_LIBRARIES_PATHS "${MATLAB_ROOT}/bin/maci")
					SET(MEX_EXTENSION mexmaci)
					SET(CMAKE_CXX_FLAGS "-m32" CACHE STRING "Flags used by the compiler during all build types." FORCE)
				ENDIF(EXISTS "${MATLAB_ROOT}/bin/maci64/")
			ENDIF(CMAKE_SIZEOF_VOID_P MATCHES "4")
		ELSE(APPLE)
  			FILE(GLOB MATLAB_PATHS "/opt/MATLAB_*")
			FOREACH(COUNT ${MATLAB_PATHS})
				IF("${COUNT}" STRGREATER "${MATLAB_PATH}")
					SET(MATLAB_PATH "${COUNT}")
				ENDIF("${COUNT}" STRGREATER "${MATLAB_PATH}")
			ENDFOREACH(COUNT ${MATLAB_PATHS})
			FIND_PATH(MATLAB_ROOT "license.txt" ${MATLAB_PATH})
			IF(NOT MATLAB_ROOT)
				MESSAGE("Linux users have to set the Matlab installation path into the MATLAB_ROOT variable.")
			ENDIF(NOT MATLAB_ROOT)
			IF(CMAKE_SIZEOF_VOID_P MATCHES "4")
	  			# 32 bits OS
				SET(MATLAB_LIBRARIES_PATHS "${MATLAB_ROOT}/bin/glnx86")
				SET(MEX_EXTENSION mexglx)
			ELSE(CMAKE_SIZEOF_VOID_P MATCHES "4")
	  			# 64 bits OS
				IF(EXISTS "${MATLAB_ROOT}/bin/glnxa64")
					SET(MATLAB_LIBRARIES_PATHS "${MATLAB_ROOT}/bin/glnxa64")
					SET(MEX_EXTENSION mexa64)
				ELSE(EXISTS "${MATLAB_ROOT}/bin/glnxa64")
					MESSAGE("You use a Matlab 32 bits version on a 64 bits OS. MPTK will be compiled in 32 bits to be useble with your Matlab version.")
					SET(MATLAB_LIBRARIES_PATHS "${MATLAB_ROOT}/bin/glnx86")
					SET(MEX_EXTENSION mexglx)
					SET(CMAKE_CXX_FLAGS "-m32" CACHE STRING "Flags used by the compiler during all build types." FORCE)
				ENDIF(EXISTS "${MATLAB_ROOT}/bin/glnxa64")
			ENDIF(CMAKE_SIZEOF_VOID_P MATCHES "4")
		ENDIF(APPLE)
	  # Common for UNIX
	  SET(MATLAB_INCLUDE_PATHS "${MATLAB_ROOT}/extern/include")
	ELSE(UNIX)
		IF(WIN32)
			SET(EXECUTABLE_EXTENSION .bat)
			SET(MATLAB_PATH 
				"[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.10;MATLABROOT]"
				"[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.9;MATLABROOT]"
				"[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.8;MATLABROOT]"
				"[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.7;MATLABROOT]"
				"[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.6;MATLABROOT]"
				"[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.5;MATLABROOT]"
				"[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.4;MATLABROOT]"
				"[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.3;MATLABROOT]"
				"[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.2;MATLABROOT]"
				"[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.1;MATLABROOT]"
			)
			FIND_PATH(MATLAB_ROOT "license.txt" ${MATLAB_PATH} NO_DEFAULT_PATH)
			IF (NOT MATLAB_ROOT)
				SET(MATLAB_PATH
					"[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.0;MATLABROOT]"
					"[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\6.5;MATLABROOT]"
		    	)
		    FIND_PATH(MATLAB_ROOT "license.txt" ${MATLAB_PATH} NO_DEFAULT_PATH)
				IF (MATLAB_ROOT)
					SET(MATLAB_OLD_WIN_MEXFILE_EXT 1 CACHE STRING "Old MEX extension for Windows")
				ENDIF (MATLAB_ROOT)
			ENDIF (NOT MATLAB_ROOT)
		  
			SET(MATLAB_LIBRARIES_PATHS
				"${MATLAB_ROOT}/extern/lib/win64/microsoft"
				"${MATLAB_ROOT}/extern/lib/win32/microsoft"
				"${MATLAB_ROOT}/extern/lib/win32/microsoft/msvc70"
				"${MATLAB_ROOT}/extern/lib/win32/microsoft/msvc71")
			SET(MATLAB_INCLUDE_PATHS "${MATLAB_ROOT}/extern/include")	
	
			# MEX files extension
			IF(CMAKE_CXX_COMPILER MATCHES "^.*cl.exe$" OR CMAKE_CXX_COMPILER MATCHES "^.*cl$")
				IF(MATLAB_OLD_WIN_MEXFILE_EXT)
					SET(MEX_EXTENSION dll)
				ELSE(MATLAB_OLD_WIN_MEXFILE_EXT)
					IF(CMAKE_CL_64)
						SET(MEX_EXTENSION mexw64)
					ELSE(CMAKE_CL_64)
						SET(MEX_EXTENSION mexw32)
					ENDIF(CMAKE_CL_64)
				ENDIF (MATLAB_OLD_WIN_MEXFILE_EXT)
			ELSE(CMAKE_CXX_COMPILER MATCHES "^.*cl.exe$" OR CMAKE_CXX_COMPILER MATCHES "^.*cl$")
				MESSAGE(FATAL_ERROR "Matlab Mex files are only supported by MS Visual Studio")
			ENDIF(CMAKE_CXX_COMPILER MATCHES "^.*cl.exe$" OR CMAKE_CXX_COMPILER MATCHES "^.*cl$")
		ENDIF(WIN32)
	ENDIF(UNIX)
	FIND_PROGRAM(MATLAB_MEX_COMPILER mex${EXECUTABLE_EXTENSION} "${MATLAB_ROOT}/bin/" NO_DEFAULT_PATH)
	FIND_LIBRARY(MATLAB_MEX_LIBRARY ${LIBMEX} ${MATLAB_LIBRARIES_PATHS} NO_DEFAULT_PATH)
	FIND_LIBRARY(MATLAB_MX_LIBRARY ${LIBMX} ${MATLAB_LIBRARIES_PATHS} NO_DEFAULT_PATH)
	FIND_LIBRARY(MATLAB_ENG_LIBRARY ${LIBENG} ${MATLAB_LIBRARIES_PATHS} NO_DEFAULT_PATH)
	FIND_PATH(MATLAB_INCLUDE_DIR "mex.h" ${MATLAB_INCLUDE_PATHS} NO_DEFAULT_PATH)
	IF(WIN32)
	  SET(MATLAB_INCLUDE_DIR \"${MATLAB_INCLUDE_DIR}\")
	ENDIF(WIN32)
ENDIF(PIPOL_IMAGE)

# This is common to UNIX and Win32:
SET(MATLAB_LIBRARIES ${MATLAB_MEX_LIBRARY} ${MATLAB_MX_LIBRARY} ${MATLAB_ENG_LIBRARY})

