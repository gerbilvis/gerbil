vole_module_name("manipulation")
vole_module_description("Various manipulations on multispectral images")
vole_module_variable("Gerbil_Manipulation")

vole_add_required_dependencies("OPENCV" "BOOST")
vole_add_optional_dependencies("BOOST_PROGRAM_OPTIONS" "BOOST_FILESYSTEM")
vole_add_required_modules("imginput" "similarity_measures")

vole_compile_library(
	"manipulation_shell"
)

vole_add_command("manipulation" "manipulation_shell.h" "vole::ManipulationShell")

vole_compile_library(
	"manipulation_config" 
)

vole_add_module()
