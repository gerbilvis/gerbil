vole_module_name("sift")
vole_module_description("SIFT features (external library)")
vole_module_variable("Vole_Sift")

vole_add_required_dependencies("OPENCV")
vole_add_required_modules()

vole_compile_library(
	"imgfeatures.cpp"
	"sift.cpp"
	"utils.cpp"
#	"dspfeat.cpp"
#	"kdtree.cpp"
#	"match.cpp"
#	"minpq.cpp"
#	"siftfeat.cpp"
#	"xform.cpp"
)

vole_add_module()
