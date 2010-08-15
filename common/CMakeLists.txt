VoleList("VOLE_MODULES_TO_BUILD" "gui")
if (VoleList_ELEMENT)
#	if (VOLE_WRAPPED_UI_FILES)
#		set(this_module_src "${VOLE_WRAPPED_UI_FILES}" "${this_module_src}")
#		VoleList("this_module_src" "${VOLE_WRAPPED_UI_FILES}" "appendList")
#	endif (VOLE_WRAPPED_UI_FILES)

	if (this_module_moc_files)
		QT4_WRAP_CPP(mocced_files ${this_module_moc_files})
		VoleList("this_module_src" "${mocced_files}" "appendList")
	endif (this_module_moc_files)

	if (this_module_rcc_files)
		QT4_ADD_RESOURCES(rcc_files ${this_module_rcc_files})
		VoleList("this_module_src" "${rcc_files}" "appendList")
	endif (this_module_rcc_files)
endif (VoleList_ELEMENT)

add_library(${this_build_library} ${this_module_src}) 

set(link_to_these_libraries)
# build list of libraries to link to 
foreach (precond ${this_module_precondition_requirements})
	VoleSymbolName(${precond} "LIBRARY_TARGETS" mod_library_targets)
	set(link_to_these_libraries ${link_to_these_libraries} ${${mod_library_targets}})
endforeach (precond ${this_module_precondition_requirements})

target_link_libraries(${this_build_library} ${link_to_these_libraries})

#target_link_libraries(${this_build_library} ${this_module_precondition_requirements})
#target_link_libraries(${this_build_library})
