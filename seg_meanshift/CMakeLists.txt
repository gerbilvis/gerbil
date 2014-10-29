vole_module_name("seg_meanshift")
vole_module_description("Fast Adaptive Mean Shift Segmentation by Georgescu et al.")
vole_module_variable("Gerbil_Seg_MeanShift")

vole_add_required_dependencies("OPENCV" "BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_FILESYSTEM" "TBB")
vole_add_required_dependencies("BOOST_THREAD" "BOOST_DATE_TIME" "BOOST_CHRONO")
vole_add_required_modules("imginput" "lsh")
vole_add_optional_modules("seg_felzenszwalb" "som")

vole_add_command("meanshift" "meanshift_shell.h" "seg_meanshift::MeanShiftShell")
vole_add_command("meanshiftsp" "meanshift_sp.h" "seg_meanshift::MeanShiftSP")
vole_add_command("meanshiftsom" "meanshift_som.h" "seg_meanshift::MeanShiftSOM")

vole_compile_library(
	"mfams" "io" "mode_pruning"
	"meanshift"         "meanshift_config"
	"meanshift_shell"
	"meanshift_sp"
	"meanshift_som"
)

vole_add_module()
