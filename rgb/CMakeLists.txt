vole_module_name("rgb")
vole_module_description("RGB true-color or false-color from multispectral images")
vole_module_variable("Gerbil_RGB")

vole_add_required_dependencies("OPENCV" "TBB")
vole_add_optional_dependencies("BOOST")
vole_add_required_modules(imginput)
vole_add_optional_modules(som)

vole_add_command("rgb" "rgb.h" "rgb::RGB")

vole_compile_library(
	"rgb"
	"rgb_config"
)

vole_add_module()
