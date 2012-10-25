vole_module_name("seg_medianshift")
vole_module_description("Median Shift segmentation by Shapira et al.")
vole_module_variable("Gerbil_Seg_MedianShift")

vole_add_required_dependencies("OPENCV")
vole_add_optional_dependencies("BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_FILESYSTEM")
vole_add_required_modules("imginput" "lsh")

vole_compile_library(
	"medianshift_shell"
)

vole_add_command("medianshift" "medianshift_shell.h" "vole::MedianShiftShell")

vole_compile_library(
	"medianshift"         "medianshift_config"
)

vole_add_module()
