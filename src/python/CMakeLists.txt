# Build python files
IF(BUILD_PYTHON_FILES)
	IF( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
		ADD_DEFINITIONS(-fPIC)
	ENDIF( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
	INCLUDE(${CMAKE_ROOT}/Modules/FindPythonLibs.cmake)
	INCLUDE(${CMAKE_ROOT}/Modules/FindPythonInterp.cmake)
	INCLUDE(${MPTK_SOURCE_DIR}/CMake/FindNumPy.cmake)
#	INCLUDE(${MPTK_SOURCE_DIR}/CMake/FindMatplotLib.cmake)
	INCLUDE(${MPTK_SOURCE_DIR}/src/libmptk/MakeRegressionConstantsH.cmake)                 
	GET_TARGET_PROPERTY(LIB_MPTK_OUTPUT_NAME mptk OUTPUT_NAME)
	IF(PYTHON_EXECUTABLE)
		SET(MPTK_PYTHON_SOURCE_DIR  ${MPTK_SOURCE_DIR}/src/python)
		SET(MPTK_PYTHON_OUTPUT_DIR  ${MPTK_BINARY_DIR}/src/python)
		SET(MPTK_PYTHON_INSTALL_DIR mptk/python)
		# TODO we should loop over filenames here
		CONFIGURE_FILE(${MPTK_PYTHON_SOURCE_DIR}/pyMPTK.h             ${MPTK_PYTHON_OUTPUT_DIR}/pyMPTK.h        COPYONLY)
		CONFIGURE_FILE(${MPTK_PYTHON_SOURCE_DIR}/pyMPTK.cpp           ${MPTK_PYTHON_OUTPUT_DIR}/pyMPTK.cpp      COPYONLY)
		CONFIGURE_FILE(${MPTK_PYTHON_SOURCE_DIR}/pyMPTK_book.cpp      ${MPTK_PYTHON_OUTPUT_DIR}/pyMPTK_book.cpp COPYONLY)
		CONFIGURE_FILE(${MPTK_PYTHON_SOURCE_DIR}/pyMPTK_atom.cpp      ${MPTK_PYTHON_OUTPUT_DIR}/pyMPTK_atom.cpp COPYONLY)
		CONFIGURE_FILE(${MPTK_PYTHON_SOURCE_DIR}/pyMPTK_decompose.cpp ${MPTK_PYTHON_OUTPUT_DIR}/pyMPTK_decompose.cpp COPYONLY)
		CONFIGURE_FILE(${MPTK_PYTHON_SOURCE_DIR}/mptkplot.py          ${MPTK_PYTHON_OUTPUT_DIR}/mptkplot.py     COPYONLY)
		CONFIGURE_FILE(${MPTK_PYTHON_SOURCE_DIR}/setup.py             ${MPTK_PYTHON_OUTPUT_DIR}/setup.py           @ONLY)
		ADD_CUSTOM_TARGET(python_file ALL echo
			DEPENDS   ${MPTK_BINARY_DIR}/src/python/pyMPTK.h
			DEPENDS   ${MPTK_BINARY_DIR}/src/python/pyMPTK.cpp
			DEPENDS   ${MPTK_BINARY_DIR}/src/python/pyMPTK_book.cpp
			DEPENDS   ${MPTK_BINARY_DIR}/src/python/pyMPTK_atom.cpp
			DEPENDS   ${MPTK_BINARY_DIR}/src/python/pyMPTK_decompose.cpp
		)
		ADD_CUSTOM_COMMAND( 
		    TARGET    python_file 
		    COMMAND   ${PYTHON_EXECUTABLE} ${MPTK_BINARY_DIR}/src/python/setup.py build
		    COMMENT   "python"
		  )
		CONFIGURE_FILE(${MPTK_SOURCE_DIR}/src/python/CMakeInstallPython.in.cmake ${MPTK_BINARY_DIR}/src/python/CMakeInstallPython.cmake @ONLY)
		INSTALL(SCRIPT ${MPTK_BINARY_DIR}/src/python/CMakeInstallPython.cmake)
	ELSE(PYTHON_EXECUTABLE)
		MESSAGE(FATAL_ERROR "You requested building for python, but python was not found")
	ENDIF(PYTHON_EXECUTABLE)

	# Example files for pyMPTK
	SET(MPTK_PYTHONSCRIPTS_SOURCE_DIR  ${MPTK_SOURCE_DIR}/src/python/examples)
	SET(MPTK_PYTHONSCRIPTS_OUTPUT_DIR  ${MPTK_BINARY_DIR}/mptk/python)
	SET(MPTK_PYTHONSCRIPTS_INSTALL_DIR mptk/python)
	FILE(GLOB PythonExampleFiles RELATIVE ${MPTK_PYTHONSCRIPTS_SOURCE_DIR} "${MPTK_PYTHONSCRIPTS_SOURCE_DIR}/*.py" "${MPTK_PYTHONSCRIPTS_SOURCE_DIR}/*.xml")
	SET(MPTK_CONFIG_FILENAME     ${CMAKE_INSTALL_PREFIX}/mptk/path.xml) # TODO - copied from matlab cmakelists but should be global
	SET(MPTK_CHIFFCHAFF_FILENAME ${CMAKE_INSTALL_PREFIX}/mptk/reference/signal/chiffchaff.wav)
	FOREACH(OnePyFile ${PythonExampleFiles})
		CONFIGURE_FILE(${MPTK_PYTHONSCRIPTS_SOURCE_DIR}/${OnePyFile} ${MPTK_PYTHONSCRIPTS_OUTPUT_DIR}/${OnePyFile} @ONLY)
		INSTALL(FILES ${MPTK_PYTHONSCRIPTS_OUTPUT_DIR}/${OnePyFile} DESTINATION ${MPTK_PYTHONSCRIPTS_INSTALL_DIR})
	ENDFOREACH()

ENDIF(BUILD_PYTHON_FILES)

