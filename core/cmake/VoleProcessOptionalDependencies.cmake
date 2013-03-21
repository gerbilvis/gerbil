set(OPTIONAL_CPP_FILE "${CMAKE_CURRENT_BINARY_DIR}/optional.cpp" )
file(WRITE ${OPTIONAL_CPP_FILE} "/* This file was created to trick CMake into creating an empty library.*/\n")

# Check dependencies
set(finished FALSE)
while(NOT finished)
	foreach(module ${VOLE_MODULE_LIST})
		list(GET ${module} 0 module_name)
		list(GET ${module} 1 module_description)
		list(GET ${module} 2 dependencies_found)
		list(GET ${module} 3 module_variable)
		list(GET ${module} 4 required_modules)
		list(GET ${module} 5 required_dependencies)
		list(GET ${module} 6 command_list)
		list(GET ${module} 7 module_ok)
		list(GET ${module} 8 optional_modules)
		list(GET ${module} 9 module_flags)

		list(LENGTH ${module_flags} old_length)

		set(${module_flags}
			""
			CACHE INTERNAL
			"Module flags"
			FORCE
		)

		string(TOUPPER ${module_name} module_name_caps)
		list(APPEND ${module_flags} "WITH_${module_name_caps}")

		foreach(required_module ${${required_modules}})
			string(TOUPPER ${required_module} required_module_name_caps)
			list(APPEND ${module_flags} "WITH_${required_module_name_caps}")

			vole_get_flags(${required_module} required_module_flags)
			list(APPEND ${module_flags} ${required_module_flags})
		endforeach()

		if(${optional_modules})
			foreach(optional_module ${${optional_modules}})
				vole_check_module(${optional_module} ok available variable)

				if(ok)
					if(VOLE_AUTO_OPTIONAL_MODULES OR ${variable})
						string(TOUPPER ${optional_module} optional_module_name_caps)
						list(APPEND ${module_flags} "WITH_${optional_module_name_caps}")

						vole_get_flags(${optional_module} optional_module_flags)
						list(APPEND ${module_flags} ${optional_module_flags})
					endif()
				endif()
			endforeach()
		endif()

		list(REMOVE_DUPLICATES ${module_flags})

		set(${module_flags}
			${${module_flags}}
			CACHE INTERNAL
			"Module flags"
			FORCE
		)

		list(LENGTH ${module_flags} new_length)

		if(new_length EQUAL old_length)
			set(finished TRUE)
		else()
			set(finished FALSE)
		endif()
	endforeach()
endwhile()

vole_debug_message("Optional module dependencies:")

foreach(module ${VOLE_MODULE_LIST})
	list(GET ${module} 0 module_name)
	list(GET ${module} 1 module_description)
	list(GET ${module} 2 dependencies_found)
	list(GET ${module} 3 module_variable)
	list(GET ${module} 4 required_modules)
	list(GET ${module} 5 required_dependencies)
	list(GET ${module} 6 command_list)
	list(GET ${module} 7 module_ok)
	list(GET ${module} 8 optional_modules)
	list(GET ${module} 9 module_flags)
	list(GET ${module} 10 module_executables)

	if(${module_ok})
		vole_debug_message("  * ${module_name}:")

		foreach(flag ${${module_flags}})
			set_property(TARGET ${module_name}-lib APPEND PROPERTY COMPILE_DEFINITIONS ${flag})
		endforeach()

		if(${optional_modules})
			set(module_name_opt "${module_name}${VOLE_OPTIONAL_LIBRARY_SUFFIX}")
			add_library(${module_name_opt} EXCLUDE_FROM_ALL ${OPTIONAL_CPP_FILE})
			# enforce consistency for IDEs that parse compile definitions
			foreach(flag ${${module_flags}})
				set_property(TARGET ${module_name_opt} APPEND PROPERTY COMPILE_DEFINITIONS ${flag})
			endforeach()

			foreach(optional_module ${${optional_modules}})
				vole_check_module(${optional_module} ok available variable)

				if(ok)
					if(VOLE_AUTO_OPTIONAL_MODULES OR ${variable})
						target_link_libraries(${module_name_opt} "${optional_module}${VOLE_LIBRARY_SUFFIX}")
						vole_debug_message("    Module \"${module_name}\" will be build with support for \"${optional_module}\".")
					else()
						vole_debug_message("    Module \"${module_name}\" will be build without support for \"${optional_module}\".")
					endif()
				else()
					vole_debug_message("    Module \"${module_name}\" will be build without support for \"${optional_module}\".")
				endif()
			endforeach()
		endif()


		foreach(executable ${${module_executables}})
			foreach(flag ${${module_flags}})
				set_property(TARGET ${executable} APPEND PROPERTY COMPILE_DEFINITIONS ${flag})
			endforeach()
		endforeach()

		vole_debug_message("    Flags: ${${module_flags}}")
	endif()
endforeach()

vole_debug_message("")
