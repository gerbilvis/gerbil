# new world order
#  cmake_minimum_required(VERSION 2.6)
#  if (vole_is_getting_cmaked)
#  	return()
#  endif(vole_is_getting_cmaked)
#  set(vole_is_getting_cmaked true)
#  
#  cmake_minimum_required(VERSION 2.6)
#  
#  #set (CMAKE_BINARY_OUTPUT_DIR "${VOLE_SOURCE_DIR}/bin/")
#  
#  project(VOLE)
#  
#  # Defined Variables for the build process:
#  # VOLE_UNIT_DIRECTORIES: Directories where components and modules are stored - entered by the user
#  # VOLE_AVAILABLE_COMPONENTS_DESCRIPTIONS: list of component description files
#  # VOLE_AVAILABLE_COMPONENTS_SYMBOLS: Symbols of the components
#  # VOLE_AVAILABLE_COMPONENTS: list of components that are available
#  # VOLE_SELECTED_COMPONENTS: list of components that are selected for single or bundle build
#  # VOLE_SELECTED_COMPONENTS_BUNDLE: list of components that should be bundled
#  # VOLE_SELECTED_COMPONENTS_SINGLE: list of components whose modules should be separately built
#  # VOLE_COMPONENT_BUNDLE_SYMBOLS: symbols for activating the "bundle" options on components
#  # VOLE_COMPONENT_SINGLE_SYMBOLS: symbols for activating the "single" options on components 
#  # VOLE_AVAILABLE_MODULES_DESCRIPTIONS: list of component/module files that are available
#  # VOLE_AVAILABLE_MODULES: list of component/module names that are available
#  # VOLE_REQUIRED_MODULES: list of all (internal) modules that are explicitly selected for building
#  # VOLE_OPTIONAL_MODULES: list of all (internal) modules that are optional
#  # VOLE_REQUIRED_EXTERNAL_MODULES: list of all external libraries, e.g. boost, opencv, that are needed
#  # VOLE_OPTIONAL_EXTERNAL_MODULES: list of all external libraries, e.g. boost, opencv, that are needed
#  # VOLE_BINARY_TARGETS_BUNDLE: all targets that should be created
#  # VOLE_BINARY_TARGETS_SINGLE: all targets that should be created
#  # VOLE_DEPENDENCIES_NO_TARGETS: all modules that should be built because targets depend on them (create no own binary targets)
#  
#  # VOLE_COMPILED_LIBS: modules that are actually compiled in libraries
#  # VOLE_COMPILED_LIBS_TARGETS: target names of the compiled libraries
#  # VOLE_COMPILED_EXECUTABLES: modules that are actually compiled in executables
#  # VOLE_COMPILED_EXECUTABLES_TARGETS: target names of the compiled executables
#  
#  # VOLE_MODULES_TO_BUILD: modules that should be build (i.e.
#  # VOLE_EXTERNAL_LIBS: modules that should be build (i.e.
#  #                         VOLE_REQUIRED_MODULES + their requirements)
#  # VOLE_MODULE_SUBDIRECTORIES: Directories that are used in add_subdirectory(..) commands
#  #   (i.e. modules that are built)
#  # VOLE_MODULES_LIBRARIES_TO_LINK: Library names of modules that should be
#  #         linked against the executables
#  # VOLE_DOXY_DEFINES_LIST: list of defines that should be expanded by doxygen
#  if(WIN32)
#  	add_definitions(-DNOMINMAX) # prevents min- and max-macro brainf*ck in VS 2003
#  endif(WIN32)


cmake_minimum_required(VERSION 2.6)
cmake_policy(SET CMP0009 NEW)
project(Vole)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(VoleHelperMacros)
include(VoleModuleMacros)

if(WIN32)
	add_definitions(-DNOMINMAX) # prevents min- and max-macro brainf*ck in VS 2003
endif(WIN32)

# Global options
option(VOLE_CMAKE_DEBUG_OUTPUT "Show additonal cmake debug output." OFF)
mark_as_advanced(VOLE_CMAKE_DEBUG_OUTPUT)
option(VOLE_CHECK_DEPENDENCIES "Do additional dependency check with nice error messages." ON)
mark_as_advanced(VOLE_CHECK_DEPENDENCIES)
option(VOLE_AUTO_OPTIONAL_MODULES "Automatically link optional modules if possible." ON)
mark_as_advanced(VOLE_AUTO_OPTIONAL_MODULES)
option(VOLE_SINGLE_TARGETS "Build single targets for each command." OFF)
set(VOLE_MODULE_DIRECTORIES "" CACHE STRING "Semicolon-separated list of module directories for vole, e.g. the forensics or the color code.")

# Global variables
set(VOLE_LIBRARY_SUFFIX "-lib" CACHE INTERNAL "Module library suffix" FORCE)
set(VOLE_OPTIONAL_LIBRARY_SUFFIX "-optional-lib" CACHE INTERNAL "Module optional library suffix" FORCE)
set(VOLE_PACKAGE_LIST "" CACHE INTERNAL "Package list" FORCE)
set(VOLE_EXTERNAL_SUBDIRECTORIES "" CACHE INTERNAL "External subdirectories" FORCE)
set(VOLE_MODULE_LIST "" CACHE INTERNAL "Module list" FORCE)

include(VoleFindPackages)
include(VoleFindExternalModules)

include(VolePythonExtension)

include_directories(common storage shell ${VOLE_EXTERNAL_SUBDIRECTORIES} "/disks/data1/riess/selbstgebaut/CGAL-3.8/include")
link_directories(${ITK_LIBRARY_DIRS})

add_subdirectory(common core/common)
add_subdirectory(storage core/storage)

foreach(dir ${VOLE_EXTERNAL_SUBDIRECTORIES})
	get_filename_component(name ${dir} NAME)
	add_subdirectory(${dir} ${name})
endforeach()

add_subdirectory(shell core/shell)

include(VoleCheckDependencies)
include(VoleProcessOptionalDependencies)


# new world order

#  set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${VOLE_SOURCE_DIR}/CMakeModules")
#  include(VoleFunctionModuleFound)
#  include(VoleList)
#  include(VoleComponentList)
#  include(VoleSymbolName)
#  include(VoleIndexValueLookup)
#  include(VoleConditionalShowModule)
#  include(VoleShellConfiguration)
#  include(VoleGuiConfiguration)
#  include(VoleShellConfigurationSingle)
#  include(VoleGuiConfigurationSingle)
#  include(VoleModuleMacros)
#  
#  
#  
#  include(DefineCompilerFlags)
#  include(VoleFindAvailableComponents)
#  include(VoleFindAvailableModules)
#  # FIXME: external dependencies are not yet considered
#  include(VoleRegisterRequiredModules)
#  include(VoleCheckExistenceOfRequiredModules)
#  include(VoleAddModuleInformation)
#  include(VoleAddExternalLibraries)
#  include(VoleUiCompileIfGui)
#  include(VoleAddSubdirectories)
#  link_directories(
#  	${VOLE_BINARY_DIR}
#  	"${VOLE_BINARY_DIR}/bin/"
#  )
#  include(VoleAddExecutableTargets)
#  include(VoleAddDependencies)

INCLUDE("cmake/TargetDoc.cmake" OPTIONAL)

