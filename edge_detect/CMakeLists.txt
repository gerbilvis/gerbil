vole_module_name(edge_detect)
vole_module_description("Edge detection in multispectral images")
vole_module_variable(Gerbil_Edge_Detect)

vole_add_required_dependencies(OPENCV)
#vole_add_required_dependencies(OpenCL)
vole_add_optional_dependencies(BOOST BOOST_PROGRAM_OPTIONS BOOST_FILESYSTEM)
vole_add_required_modules(similarity_measures imginput som)

vole_compile_library(
	edge_detection
)

vole_add_command(edge_detect edge_detection.h EdgeDetection)

vole_compile_library(
	edge_detection
	edge_detection_config
	)

vole_add_module()


