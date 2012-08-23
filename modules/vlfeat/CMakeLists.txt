vole_module_name("vlfeat")
vole_module_description("VLFeat Library (external library)")
vole_module_variable("Vole_VLFeat")

vole_add_required_dependencies("OPENCV" "BOOST" "BOOST_PROGRAM_OPTIONS")
vole_add_required_modules()

vole_compile_library(
	"quickshift_demo.cpp" "quickshift_demo.h"
)

vole_add_command("qs_demo" "quickshift_demo.h" "quickshiftDemo")

vole_compile_library(
	"vl/aib.c"         "vl/aib.h"
	"vl/dsift.c"       "vl/dsift.h"
	"vl/float.th"
	"vl/generic.c"     "vl/generic.h"
	"vl/getopt_long.c" "vl/getopt_long.h"
	"vl/heap.c"        "vl/heap.h" "vl/heap-t.h"
	"vl/host.c"        "vl/host.h"
	"vl/imop.c"        "vl/imop.h"
	"vl/imopv.c"       "vl/imopv.h"
	"vl/imopv_sse2.c"  "vl/imopv_sse2.h"
	"vl/hikmeans.c"    "vl/hikmeans.h"
	"vl/ikmeans.c"     "vl/ikmeans.h"
	"vl/kdtree.c"      "vl/kdtree.h"
	"vl/mathop.c"      "vl/mathop.h"
	"vl/mathop_sse2.c" "vl/mathop_sse2.h"
	"vl/mser.c"        "vl/mser.h"
	"vl/pgm.c"         "vl/pgm.h"
	"vl/quickshift.c"  "vl/quickshift.h"
	"vl/random.c"      "vl/random.h"
	"vl/rodrigues.c"   "vl/rodrigues.h"
	"vl/sift.c"        "vl/sift.h"
	"vl/stringop.c"    "vl/stringop.h"
	"vole_quickshift.h"
	"vole_quickshift.cpp"
	"quickshift_config.h"
	"quickshift_config.cpp"
)

vole_add_module()
