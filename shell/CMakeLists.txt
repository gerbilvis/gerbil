vole_module_name("shell")
vole_module_description("Shell frontend")
vole_module_variable("Gerbil_Shell")

vole_add_required_dependencies("BOOST" "BOOST_PROGRAM_OPTIONS")
vole_add_required_modules("core")

vole_shell_configuration(vole_include_header vole_include_command vole_modules)
configure_file(modules.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/modules.cpp)
configure_file(modules.h.in ${CMAKE_CURRENT_BINARY_DIR}/modules.h)
vole_add_required_modules(${vole_modules})

vole_compile_library(
	command.h
	${CMAKE_CURRENT_BINARY_DIR}/modules.cpp
	${CMAKE_CURRENT_BINARY_DIR}/modules.h
)

vole_add_executable("gerbil" "main.cxx")

vole_add_module()
