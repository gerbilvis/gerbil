vole_module_name("imginput")
vole_module_description("Loading and preprocessing of multi_img input")
vole_module_variable("Gerbil_ImgInput")

vole_add_required_dependencies("OPENCV" "BOOST" "BOOST_PROGRAM_OPTIONS")
vole_add_optional_dependencies("GDAL")

vole_compile_library(
	"imginput"
	"imginput_config"
	"gdalreader"
	"export"
)

vole_add_command(export export.h imginput::Export)

vole_add_module()
