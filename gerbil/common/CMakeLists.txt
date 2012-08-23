vole_module_name("gerbil_common")
vole_module_description("Common classes used in gerbil")
vole_module_variable("Gerbil_Common")

vole_add_required_dependencies("OPENCV")
vole_add_optional_dependencies("BOOST_SYSTEM" "BOOST_FILESYSTEM")

vole_compile_library(
	multi_img_ext
	multi_img_io_ext
	illuminant
)

vole_add_module()
