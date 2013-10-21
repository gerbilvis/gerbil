include(VoleHelperMacros)

macro(vole_module_name name)
	set(vole_module_name ${name})
endmacro()

macro(vole_module_description description)
	set(vole_module_description ${description})
endmacro()

macro(vole_module_variable variable)
	set(vole_module_variable ${variable})
endmacro()

macro(vole_add_required_dependencies)
	list(APPEND vole_module_required_dependencies ${ARGV})
endmacro()

macro(vole_add_optional_dependencies)
	list(APPEND vole_module_optional_dependencies ${ARGV})
endmacro()

macro(vole_add_required_modules)
	list(APPEND vole_module_required_modules ${ARGV})
endmacro()

macro(vole_add_optional_modules)
	list(APPEND vole_module_optional_modules ${ARGV})
endmacro()

macro(vole_add_include_directories)
	include_directories(${ARGV})
endmacro()

macro(vole_compile_library)
	list(APPEND vole_module_library_sources ${ARGV})
endmacro()

macro(vole_moc_library)
	list(APPEND vole_module_moc_sources ${ARGV})
endmacro()

macro(vole_ui_library)
	list(APPEND vole_module_ui_sources ${ARGV})
endmacro()

macro(vole_add_resources)
	list(APPEND vole_module_rcc_sources ${ARGV})
endmacro()

macro(vole_add_command name header class)
	set(VOLE_MODULE_COMMAND_${name}
		"${name};${header};${class}"
		CACHE INTERNAL
		"Commands"
		FORCE
	)

	list(APPEND vole_module_commands VOLE_MODULE_COMMAND_${name})
endmacro()

macro(vole_add_executable name)
	if(vole_module_executable_${name})
		message(FATAL_ERROR "Executable \"${name} added in ${CMAKE_CURRENT_SOURCE_DIR} already exists!")
	endif()

	list(APPEND vole_module_executable_${name} ${name} ${ARGN})
	list(APPEND vole_module_executables vole_module_executable_${name})
endmacro()

macro(vole_add_python_module name file)
  if (NOT Boost_PYTHON_FOUND)
    vole_debug_message("Python module ${name} will be unavailable due to missing Boost.Python.")
  else()
    list(APPEND vole_python_module_${name} ${name} ${file})
    list(APPEND vole_python_modules vole_python_module_${name})
  endif()
endmacro()

macro(vole_add_module)
	# Check if vole_module_name is set
	if(NOT vole_module_name)
		message(FATAL_ERROR "Variable vole_module_name not set in ${CMAKE_CURRENT_SOURCE_DIR}!")
		return()
	endif()

	# Check if vole_module_description is set
	if(NOT vole_module_description)
		message(FATAL_ERROR "Variable vole_module_description not set in ${CMAKE_CURRENT_SOURCE_DIR}!")
		return()
	endif()

	# Check if vole_module_variable is set
	if(NOT vole_module_variable)
		message(FATAL_ERROR "Variable vole_module_variable not set in ${CMAKE_CURRENT_SOURCE_DIR}!")
		return()
	endif()

	# Debug output
	vole_debug_message("Adding module \"${vole_module_name}\" in \"${CMAKE_CURRENT_SOURCE_DIR}\".")
	vole_debug_message("  vole_module_name: ${vole_module_name}")
	vole_debug_message("  vole_module_description: ${vole_module_description}")
	vole_debug_message("  vole_module_variable: ${vole_module_variable} = ${${vole_module_variable}}")
	vole_debug_message("  vole_module_required_dependencies: ${vole_module_required_dependencies}")
	vole_debug_message("  vole_module_required_modules: ${vole_module_required_modules}")
	vole_debug_message("  vole_module_optional_modules: ${vole_module_optional_modules}")
	vole_debug_message("  vole_module_library_sources: ${vole_module_library_sources}")
	vole_debug_message("  vole_module_moc_sources: ${vole_module_moc_sources}")
	vole_debug_message("  vole_module_ui_sources: ${vole_module_ui_sources}")
	vole_debug_message("  vole_module_rcc_sources: ${vole_module_rcc_sources}")
	vole_debug_message("  vole_module_command_names: ${vole_module_command_names}")
	vole_debug_message("  vole_module_command_header: ${vole_module_command_header}")
	vole_debug_message("  vole_module_command_classes: ${vole_module_command_classes}")
	vole_debug_message("  vole_module_executable_names: ${vole_module_executable_names}")
	vole_debug_message("  vole_module_executable_sources: ${vole_module_executable_sources}")

	# Add switch to turn build on/off
	option(${vole_module_variable} ${vole_module_description} ON)

	# Enable sse2 (#FIXME evil hack for gerbil, requires EnableSSE2.cmake)
	enable_sse2()

	# Check required dependencies
	set(required_dependencies_found TRUE)
	foreach(dependency ${vole_module_required_dependencies})
		if(NOT ${dependency}_FOUND)
			set(required_dependencies_found FALSE)
		endif()
	endforeach()

	if(required_dependencies_found)
		include_directories(${CMAKE_CURRENT_BINARY_DIR})

		# Wrap Qt sources
		if (WITH_QT5)
			qt5_wrap_cpp(moc_sources ${vole_module_moc_sources})
			qt5_wrap_ui(uic_sources ${vole_module_ui_sources})

			list(APPEND vole_module_library_sources ${moc_sources} ${uic_sources})
		elseif (WITH_QT)
			# The custom define is a workaround for faulty moc preprocessor
			# the define will disable code in boost headers that trigger the bug
			qt4_wrap_cpp(moc_sources ${vole_module_moc_sources} OPTIONS -DBOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
			qt4_wrap_ui(uic_sources ${vole_module_ui_sources})

			list(APPEND vole_module_library_sources ${moc_sources} ${uic_sources})
		endif()

		# Add library target
		set(vole_module_library "${vole_module_name}${VOLE_LIBRARY_SUFFIX}")
		if(vole_module_library_sources)
			vole_debug_message("  Adding library \"${vole_module_library}\" with sources \"${vole_module_library_sources}\".")

			add_library(${vole_module_library} EXCLUDE_FROM_ALL ${vole_module_library_sources})

			foreach(dependency ${vole_module_required_dependencies})
				target_link_libraries(${vole_module_library} ${${dependency}_LIBRARIES})
			endforeach()

			foreach(dependency ${vole_module_optional_dependencies})
				if(${dependency}_FOUND)
					target_link_libraries(${vole_module_library} ${${dependency}_LIBRARIES})
				endif()
			endforeach()

			foreach(module ${vole_module_required_modules})
				target_link_libraries(${vole_module_library} "${module}${VOLE_LIBRARY_SUFFIX}")
			endforeach()

			target_link_libraries(${vole_module_library} "core${VOLE_LIBRARY_SUFFIX}")

			if(vole_module_optional_modules)
				target_link_libraries(${vole_module_library} "${vole_module_name}${VOLE_OPTIONAL_LIBRARY_SUFFIX}")
			endif()
		endif()

		# Add python modules
		if(VOLE_BUILD_PYTHON_MODULES)
		  if(vole_python_modules)
		    add_custom_target(add_py_module_${vole_module_name} ALL ${CMAKE_COMMAND} -E echo "import ${vole_module_name}" >> ${CMAKE_BINARY_DIR}/pyvole/__init__.py COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/pyvole/${vole_module_name}/__init__.py)
		    add_dependencies(add_py_module_${vole_module_name} clean_python_module_list)
		    foreach(pymodule ${vole_python_modules})
		      list(GET ${pymodule} 0 pymodule_name)
		      list(GET ${pymodule} 1 pymodule_sources)
		      
		      if(pymodule_name AND pymodule_sources)
			vole_debug_message("adding py module ${pymodule_name} with sources ${pymodule_sources}")
			add_library(${pymodule_name} SHARED ${pymodule_sources})
			add_dependencies(${pymodule_name} clean_python_module_list)
			set_target_properties(${pymodule_name} PROPERTIES PREFIX "")
			set_target_properties(${pymodule_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/pyvole/${vole_module_name}/)
			target_link_libraries(${pymodule_name} ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARIES} ${PYTHON_LIBRARY} ${vole_module_library})
			add_custom_command(TARGET add_py_module_${vole_module_name} POST_BUILD COMMAND ${CMAKE_COMMAND} -E echo "from ${pymodule_name} import *" >> ${CMAKE_BINARY_DIR}/pyvole/${vole_module_name}/__init__.py)
		      endif()
		    endforeach()
		  endif()
		endif()

		# Add executable targets
		if(${vole_module_variable})
			foreach(executable ${vole_module_executables})
				list(GET ${executable} 0 executable_name)
				list(GET ${executable} 1 executable_sources)

				if(executable_name AND executable_sources)
					vole_debug_message("  Adding executable \"${executable_name}\" with sources \"${executable_sources}\".")

					set(rcc_sources)
					if(WITH_QT5)
						qt5_add_resources(rcc_sources ${vole_module_rcc_sources})
					elseif (WITH_QT)
						qt4_add_resources(rcc_sources ${vole_module_rcc_sources})
					endif()

					add_executable(${executable_name} ${executable_sources} ${rcc_sources})
					target_link_libraries(${executable_name} "core${VOLE_LIBRARY_SUFFIX}")
					target_link_libraries(${executable_name} ${vole_module_library})
				endif()
			endforeach()
		endif()

		# Add single shell executable targets
		if(VOLE_SINGLE_TARGETS AND ${vole_module_variable})
			foreach(command ${vole_module_commands})
				list(GET ${command} 0 command_name)
				list(GET ${command} 1 command_header)
				list(GET ${command} 2 command_class)

				set(single_target_sources
					${CMAKE_SOURCE_DIR}/shell/main.cxx
					${CMAKE_SOURCE_DIR}/shell/command.h
					${CMAKE_CURRENT_BINARY_DIR}/${command_name}/${command_name}_modules.cpp
					${CMAKE_SOURCE_DIR}/shell/modules.h
					${CMAKE_CURRENT_BINARY_DIR}/${command_name}/modules.h
				)

				set(vole_include_header "/* ${vole_module_description} */\n#include \"${command_header}\"")
				set(vole_include_command "/* ${vole_module_description} */\n	c = new ${command_class}(); insert(std::make_pair(c->getName(), c));")
#				set(vole_include_header "${vole_include_header}\n/* ${vole_module_description} */\n#include \"${command_header}\"")
#				set(vole_include_command "${vole_include_command}\n	/* ${vole_module_description} */\n	c = new ${command_class}(); insert(std::make_pair(c->getName(), c));")

				configure_file(${CMAKE_SOURCE_DIR}/shell/modules.cpp.in ${command_name}/${command_name}_modules.cpp)
				configure_file(${CMAKE_SOURCE_DIR}/shell/modules.h.in ${command_name}/modules.h)

				include_directories(${CMAKE_CURRENT_BINARY_DIR}/${command_name})
				add_definitions(-DSINGLE)

				add_executable(${command_name} ${single_target_sources})
				target_link_libraries(${command_name} ${vole_module_library})
			endforeach()
		endif()
	endif()

	# Append to global modules list
	set(VOLE_MODULE_${vole_module_name}_REQUIRED_MODULES
		${vole_module_required_modules}
		CACHE INTERNAL
		"Required modules"
		FORCE
	)

	set(VOLE_MODULE_${vole_module_name}_REQUIRED_DEPENDENCIES
		${vole_module_required_dependencies}
		CACHE INTERNAL
		"Required modules"
		FORCE
	)

	set(VOLE_MODULE_${vole_module_name}_COMMANDS
		${vole_module_commands}
		CACHE INTERNAL
		"Commands"
		FORCE
	)

	set(VOLE_MODULE_${vole_module_name}_OK
		${required_dependencies_found}
		CACHE INTERNAL
		"Commands"
		FORCE
	)

	set(VOLE_MODULE_${vole_module_name}_OPTIONAL_MODULES
		${vole_module_optional_modules}
		CACHE INTERNAL
		"Optional modules"
		FORCE
	)

	set(VOLE_MODULE_${vole_module_name}_FLAGS
		""
		CACHE INTERNAL
		"Module flags"
		FORCE
	)

	set(VOLE_MODULE_${vole_module_name}_EXECUTABLES
		""
		CACHE INTERNAL
		"Module executables"
		FORCE
	)
	if(${vole_module_variable})
		foreach(executable ${vole_module_executables})
			list(GET ${executable} 0 executable_name)
			list(GET ${executable} 1 executable_sources)

			if(executable_name AND executable_sources)
				list(APPEND VOLE_MODULE_${vole_module_name}_EXECUTABLES ${executable_name})
			endif()
		endforeach()
	endif()
	if(VOLE_SINGLE_TARGETS AND ${vole_module_variable})
		foreach(command ${vole_module_commands})
			list(GET ${command} 0 command_name)

			list(APPEND VOLE_MODULE_${vole_module_name}_EXECUTABLES ${command_name})
		endforeach()
	endif()
	set(VOLE_MODULE_${vole_module_name}_EXECUTABLES
		${VOLE_MODULE_${vole_module_name}_EXECUTABLES}
		CACHE INTERNAL
		"Module executables"
		FORCE
	)

	set(VOLE_MODULE_${vole_module_name}_DATA
		""
		CACHE INTERNAL
		"Module data"
		FORCE
	)
	list(APPEND VOLE_MODULE_${vole_module_name}_DATA
		${vole_module_name}
		${vole_module_description}
		${required_dependencies_found}
		${vole_module_variable}
		VOLE_MODULE_${vole_module_name}_REQUIRED_MODULES
		VOLE_MODULE_${vole_module_name}_REQUIRED_DEPENDENCIES
		VOLE_MODULE_${vole_module_name}_COMMANDS
		VOLE_MODULE_${vole_module_name}_OK
		VOLE_MODULE_${vole_module_name}_OPTIONAL_MODULES
		VOLE_MODULE_${vole_module_name}_FLAGS
		VOLE_MODULE_${vole_module_name}_EXECUTABLES
	)
	set(VOLE_MODULE_${vole_module_name}_DATA
		${VOLE_MODULE_${vole_module_name}_DATA}
		CACHE INTERNAL
		"Module data"
		FORCE
	)

	set(VOLE_MODULE_LIST
		"${VOLE_MODULE_LIST};VOLE_MODULE_${vole_module_name}_DATA"
		CACHE INTERNAL
		"Modules list"
		FORCE
	)

	vole_debug_message(" ")
endmacro()
