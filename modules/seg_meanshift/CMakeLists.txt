vole_module_name("seg_meanshift")
vole_module_description("Fast Adaptive Mean Shift Segmentation by Georgescu et al.")
vole_module_variable("Vole_Seg_MeanShift")

vole_add_required_dependencies("OPENCV" "BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_FILESYSTEM" "BOOST_THREAD" "BOOST_DATE_TIME" "TBB")
vole_add_required_modules("imginput" "lsh")
vole_add_optional_modules("seg_felzenszwalb2")

vole_compile_library(
	"meanshift_shell"
)

vole_add_command("meanshift" "meanshift_shell.h" "vole::MeanShiftShell")
vole_add_command("meanshiftsp" "meanshift_sp.h" "vole::MeanShiftSP")

vole_compile_library(
	"io" "mfams"
	"meanshift"         "meanshift_config"
	"meanshift_sp"
)

vole_add_module()
