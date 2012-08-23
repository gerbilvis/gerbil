vole_module_name("common")
vole_module_description("Common classes used in the vole framework")
vole_module_variable("Vole_Common")

vole_add_required_dependencies()
vole_add_optional_dependencies("BOOST_PROGRAM_OPTIONS" "QT" "OPENCV" "BOOST_PYTHON" "PYTHON" "NUMPY")
vole_add_optional_modules("gerbil_common")

vole_compile_library(
    "hashes"
    "vole_config"
    "filesystem_helpers"
    "data_traits.h"
    "global_defines.h"
    "array1d.h" "array2d.h"
    "image_group.h" "image_list.h"
    "string_conversion"
    "vole_storage"
    "vole_error.h" "vole_exception.h" "vole_image.h"
#    vole_limits.h     vole_limits.cxx
    vole_user_paths.h vole_user_paths.cpp
	vole_statistics.h vole_statistics.cpp
    "qtopencv"
    "multi_img"
    "multi_img_io"
    "labeling"
    "stopwatch"
    "progress_observer"
)

vole_add_python_module("_common" "pyvole_common.cpp")

vole_add_module()
