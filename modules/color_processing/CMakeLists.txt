vole_module_name("color_processing")
vole_module_description("Handy operations on color images")
vole_module_variable("Vole_Color_Proc")

vole_add_required_dependencies("OPENCV")
vole_add_required_modules()

vole_compile_library(
	"pixel_operations.cpp"
	"pixel_operations.txx"
	"pixel_operations.h"
)

vole_add_module()
