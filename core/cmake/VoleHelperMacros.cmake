# Show extra debug messages if option VOLE_CMAKE_DEBUG_OUTPUT is turned on
macro(vole_debug_message)
	if(VOLE_CMAKE_DEBUG_OUTPUT)
		message("${ARGV}")
	endif()
endmacro()

# Check if package was found and set include directories and definitions
macro(vole_check_package package name help_message found includes libraries)
	list(APPEND VOLE_PACKAGE_LIST ${package})
	set(VOLE_PACKAGE_LIST ${VOLE_PACKAGE_LIST} CACHE INTERNAL "Package list" FORCE)

	set(${package}_HELP_MESSAGE "${help_message}")

	if(${found})
		set(${package}_FOUND TRUE)
		set(${package}_INCLUDE_DIRS ${includes})
		set(${package}_LIBRARIES ${libraries})

		set(WITH_${package} TRUE)
		add_definitions(-DWITH_${package})
		include_directories(${includes})

		vole_debug_message("${name} found!")
		vole_debug_message("  ${name} help message: ${help_message}")
		vole_debug_message("  ${name} include directories: ${includes}")
		vole_debug_message("  ${name} libraries: ${libraries}")
		vole_debug_message(" ")
	else()
		vole_debug_message("${name} not found!")
	endif()
endmacro()

# Check module
macro(vole_check_module check_name check_ok check_available check_variable)
	set(${check_ok} FALSE)
	set(${check_available} FALSE)
	set(${check_variable} FALSE)

	foreach(check_module ${VOLE_MODULE_LIST})
		list(GET ${check_module} 0 check_module_name)
		list(GET ${check_module} 3 check_module_variable)
		list(GET ${check_module} 7 check_module_ok)

		if(${check_name} STREQUAL ${check_module_name})
			set(${check_available} TRUE)
			set(${check_ok} ${${check_module_ok}})
			set(${check_variable} ${check_module_variable})
			break()
		endif()
	endforeach()
endmacro()

# Get module flags
macro(vole_get_flags flags_name flags_flags)
	set(${flags_flags})

	foreach(flags_module ${VOLE_MODULE_LIST})
		list(GET ${flags_module} 0 flags_module_name)
		list(GET ${flags_module} 9 flags_module_flags)

		if(${flags_name} STREQUAL ${flags_module_name})
			set(${flags_flags} ${${flags_module_flags}})
			break()
		endif()
	endforeach()
endmacro()

# Generate shell configuration
macro(vole_shell_configuration include_header include_command modules)
	set(${include_header})
	set(${include_command})
	set(${modules})

	foreach(module ${VOLE_MODULE_LIST})
		list(GET ${module} 0 module_name)
		list(GET ${module} 1 module_description)
		list(GET ${module} 2 dependencies_found)
		list(GET ${module} 3 module_variable)
		list(GET ${module} 4 required_modules)
		list(GET ${module} 5 required_dependencies)
		list(GET ${module} 6 command_list)
		list(GET ${module} 7 module_ok)

		if(${module_variable} AND ${command_list})
			list(APPEND ${modules} ${module_name})

			foreach(command ${${command_list}})
				list(GET ${command} 0 command_name)
				list(GET ${command} 1 command_header)
				list(GET ${command} 2 command_class)

				set(${include_header} "${${include_header}}\n/* ${module_description} */\n#include \"${command_header}\"")
				set(${include_command} "${${include_command}}\n	/* ${module_description} */\n	c = new ${command_class}(); insert(std::make_pair(c->getName(), c));")
			endforeach()
		endif()
	endforeach()
endmacro()
