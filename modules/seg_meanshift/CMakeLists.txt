vole_module_name("seg_meanshift")
vole_module_description("Fast Adaptive Mean Shift Segmentation by Georgescu et al.")
vole_module_variable("Vole_Seg_MeanShift")

vole_add_required_dependencies("OPENCV" "BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_FILESYSTEM")
vole_add_required_modules("imginput" "lsh")

vole_compile_library(
	"meanshift_shell"
)

vole_add_command("meanshift" "meanshift_shell.h" "vole::MeanShiftShell")

vole_compile_library(
	"io" "auxiliary"
	"mfams"
	"meanshift"         "meanshift_config"
)

vole_add_module()
