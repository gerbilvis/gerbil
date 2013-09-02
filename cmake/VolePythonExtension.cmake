# Python Extension for Vole

option(VOLE_BUILD_PYTHON_MODULES "Build python modules for commands where possible." OFF)

set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "__init__.py;__init__.pyc")

if(VOLE_BUILD_PYTHON_MODULES)
  if(NOT Boost_PYTHON_FOUND OR NOT PYTHONLIBS_FOUND)
    message(FATAL_ERROR "Boost.Python and Pythonlibs are needed for building Vole Python modules. Please install them!")
  endif()

  add_custom_target(clean_python_module_list ALL COMMAND  ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/pyvole/__init__.py ${CMAKE_BINARY_DIR}/pyvole/__init__.pyc)
endif()
