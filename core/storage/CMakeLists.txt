vole_module_name("cache")
vole_module_description("Caching Super Class")
vole_module_variable("Vole_Cache")

vole_add_required_dependencies("OPENCV" "BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_SERIALIZATION" "BOOST_FILESYSTEM" "BOOST_SYSTEM")
vole_add_required_modules()

vole_compile_library(
	command_cache_adm.cpp
	command_cache_adm.h
	vole_cache_config.cpp
	vole_cache_config.h
)

vole_add_command("cache" "command_cache_adm.h" "vole::CommandCacheAdministration")

vole_compile_library(
	vole_cache_item.cpp
	vole_cache_item.h
)

vole_add_module()
