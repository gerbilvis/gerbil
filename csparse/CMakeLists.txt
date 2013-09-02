vole_module_name("csparse")
vole_module_description("A Concise Sparse Matrix Package in C (sc.fsu.edu)")
vole_module_variable("Gerbil_CSparse")

vole_compile_library(
	"cs_add.c"       "cs_fkeep.c"   "cs_multiply.c"  "cs_tdfs.c"
	"cs_amd.c"       "cs_ipvec.c"   "cs_permute.c"   "cs_transpose.c"
	"cs_compress.c"  "cs_leaf.c"    "cs_post.c"      "cs_usolve.c"
	"cs_counts.c"    "cs_lsolve.c"  "cs_reach.c"     "cs_util.c"
	"cs_cumsum.c"    "cs_lu.c"      "cs_scatter.c"
	"cs_dfs.c"       "cs_lusol.c"   "cs_spsolve.c"
	"cs_etree.c"     "cs_malloc.c"  "cs_sqr.c"
	"cs_print.c"     "cs_norm.c"
)

vole_add_module()
