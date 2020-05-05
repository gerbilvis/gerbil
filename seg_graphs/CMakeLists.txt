vole_module_name("seg_graphs")
vole_module_description("Graph Cut Segmentation by Grady et al.")
vole_module_variable("Gerbil_Seg_Graphs")

vole_add_required_dependencies("OPENCV")
vole_add_optional_dependencies("BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_FILESYSTEM")
vole_add_required_modules(similarity_measures imginput)
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
	"csparse/cs_add.c"       "csparse/cs_fkeep.c"   "csparse/cs_multiply.c"  "csparse/cs_tdfs.c"
	"csparse/cs_amd.c"       "csparse/cs_ipvec.c"   "csparse/cs_permute.c"   "csparse/cs_transpose.c"
	"csparse/cs_compress.c"  "csparse/cs_leaf.c"    "csparse/cs_post.c"      "csparse/cs_usolve.c"
	"csparse/cs_counts.c"    "csparse/cs_lsolve.c"  "csparse/cs_reach.c"     "csparse/cs_util.c"
	"csparse/cs_cumsum.c"    "csparse/cs_lu.c"      "csparse/cs_scatter.c"
	"csparse/cs_dfs.c"       "csparse/cs_lusol.c"   "csparse/cs_spsolve.c"
	"csparse/cs_etree.c"     "csparse/cs_malloc.c"  "csparse/cs_sqr.c"
	"csparse/cs_print.c"     "csparse/cs_norm.c"
)

vole_add_module()
