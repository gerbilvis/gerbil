vole_module_name("seg_fh")
vole_module_description("Superpixel Segmentation by Felzenszwalb and Huttenlocher")
vole_module_variable("Vole_Seg_FH")

vole_add_required_dependencies("OPENCV" "BOOST" "BOOST_PROGRAM_OPTIONS")
vole_add_required_modules("color_processing")

vole_compile_library(
	"command_seg_felzenszwalb.cpp" "command_seg_felzenszwalb.h"
	"fh_convolve.cpp"                 "fh_convolve.h"
	"fh_filter.cpp"                   "fh_filter.h"
	"fh_imconv.cpp"                   "fh_imconv.h"
)

vole_add_command("fhseg" "command_seg_felzenszwalb.h" "vole::CommandSegFelzenszwalb")

vole_compile_library(
	"fh_segmentation.cxx"         "fh_segmentation.h"
	"seg_felzenszwalb_config.cxx" "seg_felzenszwalb_config.h"
)

vole_add_module()
