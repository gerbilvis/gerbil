vole_module_name("computational_geometry")
vole_module_description("Geometric routines like convex hull or minimum enclosing rectangle")
vole_module_variable("Vole_CompGeometry")

vole_add_required_dependencies("OPENCV" "BOOST" "BOOST_PROGRAM_OPTIONS")
vole_add_required_modules()

vole_compile_library(
	"graham_scan.h"                 "graham_scan.cpp"
	"minimum_enclosing_rectangle.h" "minimum_enclosing_rectangle.cpp"
	"morphological_operators.h"     
	"geometry_utilities.cpp"        "geometry_utilities.h"
	"connected_components.h"        "connected_components.cpp"
)

vole_add_module()
