cmake_minimum_required(VERSION 2.6.2) # for VERSION_GREATER, lol
cmake_policy(SET CMP0009 NEW)

if(CMAKE_VERSION VERSION_GREATER "2.8.11.100")
	cmake_policy(SET CMP0022 NEW)
endif()
if(CMAKE_VERSION VERSION_GREATER "3.0")
	cmake_policy(SET CMP0038 NEW)
endif()

project(Gerbil)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# enable warnings (gcc and msvc)
if(MSVC)
	# Force to always compile with W4
	if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
		string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
	endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
	# Update if necessary
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -Wno-reorder -pedantic -std=c++11")
endif()

include(VoleHelperMacros)
include(VoleModuleMacros)

include(EnableSSE2)

# Global options
option(GERBIL_CUDA "Enable CUDA support (OpenCV < 3.0)." OFF)
if(GERBIL_CUDA)
    add_definitions(-DGERBIL_CUDA)
endif()
option(VOLE_CMAKE_DEBUG_OUTPUT "Show additonal cmake debug output." OFF)
#mark_as_advanced(VOLE_CMAKE_DEBUG_OUTPUT)
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

