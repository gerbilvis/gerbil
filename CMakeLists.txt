cmake_minimum_required(VERSION 2.6)
cmake_policy(SET CMP0009 NEW)
cmake_policy(SET CMP0022 NEW)
project(Vole)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(VoleHelperMacros)
include(VoleModuleMacros)

if(WIN32)
	add_definitions(-DNOMINMAX) # prevents min- and max-macro brainf*ck in VS 2003
endif(WIN32)

include(EnableSSE2)

# Global options
option(VOLE_CMAKE_DEBUG_OUTPUT "Show additonal cmake debug output." OFF)
mark_as_advanced(VOLE_CMAKE_DEBUG_OUTPUT)
option(VOLE_CHECK_DEPENDENCIES "Do additional dependency check with nice error messages." ON)
mark_as_advanced(VOLE_CHECK_DEPENDENCIES)
option(VOLE_AUTO_OPTIONAL_MODULES "Automatically link optional modules if possible." ON)
mark_as_advanced(VOLE_AUTO_OPTIONAL_MODULES)
option(VOLE_SINGLE_TARGETS "Build single targets for each command." OFF)
set(VOLE_MODULE_DIRECTORIES "${CMAKE_SOURCE_DIR}/" CACHE STRING "Semicolon-separated list of module directories.")

# Global variables
set(VOLE_LIBRARY_SUFFIX "-lib" CACHE INTERNAL "Module library suffix" FORCE)
set(VOLE_OPTIONAL_LIBRARY_SUFFIX "-optional-lib" CACHE INTERNAL "Module optional library suffix" FORCE)
set(VOLE_PACKAGE_LIST "" CACHE INTERNAL "Package list" FORCE)
set(VOLE_EXTERNAL_SUBDIRECTORIES "" CACHE INTERNAL "External subdirectories" FORCE)
set(VOLE_MODULE_LIST "" CACHE INTERNAL "Module list" FORCE)

include(VoleFindPackages)
include(VoleFindExternalModules)

include(VolePythonExtension)

include_directories(core shell ${VOLE_EXTERNAL_SUBDIRECTORIES})
# TODO: why?
#link_directories(${ITK_LIBRARY_DIRS})

foreach(dir ${VOLE_EXTERNAL_SUBDIRECTORIES})
	get_filename_component(name ${dir} NAME)
	string(COMPARE NOTEQUAL ${name} shell NOTSHELL)
	if (${NOTSHELL})
		# build path consists of directory name and containing directory name
		# to allow non-unique directory names (e.g. vole/common, gerbil/common)
		get_filename_component(path ${dir} PATH)
		get_filename_component(prefix ${path} NAME)
		add_subdirectory(${dir} "${prefix}/${name}")
	endif()
endforeach()

# include shell at the end when all commands are ready
add_subdirectory(shell shell)

include(VoleCheckDependencies)
include(VoleProcessOptionalDependencies)

include("cmake/TargetDoc.cmake" OPTIONAL)

