vole_module_name("seg_felzenszwalb")
vole_module_description("Felzenszwalb segmentation on multispectral images")
vole_module_variable("Gerbil_Seg_Felzenszwalb")

vole_add_required_dependencies("OPENCV")
vole_add_optional_dependencies("BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_FILESYSTEM")
vole_add_required_modules("similarity_measures")

vole_compile_library(
	"felzenszwalb_shell"
)

vole_add_command("felzenszwalb" "felzenszwalb_shell.h" "gerbil::FelzenszwalbShell")

vole_compile_library(
	"felzenszwalb"
	"segment" "graph"
	"felzenszwalb2_config"
)

vole_add_module()
