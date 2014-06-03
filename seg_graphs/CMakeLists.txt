vole_module_name("seg_graphs")
vole_module_description("Graph Cut Segmentation by Grady et al.")
vole_module_variable("Gerbil_Seg_Graphs")

vole_add_required_dependencies("OPENCV")
vole_add_optional_dependencies("BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_FILESYSTEM")
vole_add_required_modules(csparse similarity_measures imginput)
vole_add_optional_modules(som)

vole_compile_library(
	"graphseg_shell"
)

vole_add_command("graphseg" "graphseg_shell.h" "seg_graphs::GraphSegShell")

vole_compile_library(
	"geodesic.cpp"    "sorting"
	"random_walker.cpp"  "spanning_forest.cpp"
	"graph"
	"graph_alg.h"
	"graphseg"         "graphseg_config"
)

vole_add_module()
