foreach(dir ${VOLE_MODULE_DIRECTORIES})
	vole_debug_message("Searching for external modules in \"${dir}\".")

	if(NOT IS_DIRECTORY ${dir})
		set(dir "${CMAKE_SOURCE_DIR}/${dir}")
	endif()

	get_filename_component(path ${dir} ABSOLUTE)
	get_filename_component(path_name ${path} NAME)

	file(GLOB found_cmake_lists "${path}/CMakeLists.txt")

	if(found_cmake_lists)
		if(NOT ${path} STREQUAL ${CMAKE_SOURCE_DIR})
			vole_debug_message("Adding subdirectory \"${path}\".")
			vole_debug_message(" ")

			list(APPEND VOLE_EXTERNAL_SUBDIRECTORIES ${path})
		else()
			vole_debug_message("Cannot add vole source directory as module directory!")
			vole_debug_message(" ")
		endif()
	else()
		vole_debug_message("Could not find CMakeLists.txt in directory \"${path}\"!")
		vole_debug_message("Searching for CMakeLists.txt in subdirectories.")

		file(GLOB_RECURSE found_cmake_lists_subdirs TRUE "${path}/*/CMakeLists.txt")

		if(found_cmake_lists_subdirs)
			foreach(cmake_lists ${found_cmake_lists_subdirs})
				# *first* make the path of filename absolute, then cut off the filename
				# the other way round cmake fails with Windows network shares
				get_filename_component(path_absolute ${cmake_lists} ABSOLUTE)
				get_filename_component(path_subdir ${path_absolute} PATH)
				get_filename_component(path_name_subdir ${path_subdir} NAME)

				if(NOT ${path_subdir} STREQUAL ${CMAKE_SOURCE_DIR})
					vole_debug_message("Adding subdirectory \"${path_subdir}\".")
					vole_debug_message(" ")

					list(APPEND VOLE_EXTERNAL_SUBDIRECTORIES ${path_subdir})
				else()
					vole_debug_message("Cannot add vole source directory as module directory!")
					vole_debug_message(" ")
				endif()
			endforeach()
		else()
			vole_debug_message("Could not find any modules in subdirectories of \"${path}\", skipped!")
		endif()
	endif()
endforeach()
