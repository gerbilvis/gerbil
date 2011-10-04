vole_module_name("ms_edges")
vole_module_description("Edge detection in multispectral images")
vole_module_variable("Vole_MSedges")

vole_add_required_dependencies("OPENCV" "BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_SYSTEM")
vole_add_optional_modules("gerbil_common")
vole_add_required_modules("similarity_measures")

vole_compile_library(
	"edge_detection"
)

vole_add_command("edge_detect" "edge_detection.h" "EdgeDetection")

vole_compile_library(
	"edge_detection_config"
	"myCanny"
	"_cvgeom.h"
	"_cvimgproc.h"
	"self_organizing_map"
	"neuron"
	"som_trainer"
	"space_filling_curve"
	"Graph/completingGraph"
	"Graph/graph"
	"Graph/misc"
	"Graph/solution"
	"Graph/dijkstra"
	"Graph/smallWorldGraph"
  "Graph/fastDijkstra"
  "GTM/rbfnet"
  "GTM/gmm"
  "GTM/gtm"
  "GTM/gtm_em"
  "GTM/gtm_defs.h"
  "GTM/gtm_misc"
)

vole_add_module()
