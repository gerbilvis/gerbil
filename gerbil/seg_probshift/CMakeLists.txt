vole_module_name("seg_probshift")
vole_module_description("Probabilistic Shift segmentation by Shetty and Ahuja")
vole_module_variable("Gerbil_Seg_ProbShift")

vole_add_required_dependencies("OPENCV" "BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_FILESYSTEM" "PETSC" "SLEPC")
vole_add_required_modules("imginput" "lsh" "seg_meanshift")

vole_compile_library(
	"probshift_shell"
)

vole_add_command("probshift" "probshift_shell.h" "vole::ProbShiftShell")

vole_compile_library(
	"probshift"
	"probshift_config"
	"evrot"
)

vole_add_module()
