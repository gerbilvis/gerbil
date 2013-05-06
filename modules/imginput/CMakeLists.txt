vole_module_name("imginput")
vole_module_description("Loading and preprocessing of multi_img input")
vole_module_variable("Vole_ImgInput")

vole_add_required_dependencies("OPENCV" "BOOST" "BOOST_PROGRAM_OPTIONS")
vole_add_optional_dependencies("GDAL")
vole_add_optional_modules("gerbil_common")

vole_compile_library(
	"imginput"
	"imginput_config"
        "gdalreader"
)

vole_add_module()
