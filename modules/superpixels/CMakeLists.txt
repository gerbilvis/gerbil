vole_module_name("superpixels")
vole_module_description("Algorithms for Superpixel Segmentation")
vole_module_variable("Vole_Superpixels")

vole_add_required_dependencies("OPENCV" "BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_SERIALIZATION")
vole_add_required_modules("cache" "rbase")

vole_compile_library(
	commands/command_felzenszwalb.cpp
	commands/command_felzenszwalb.h
	commands/felzenszwalb_config.cpp
	commands/felzenszwalb_config.h

	commands/command_veksler.cpp
	commands/command_veksler.h
	commands/veksler_config.cpp
	commands/veksler_config.h

	commands/patches_config.cpp
	commands/patches_config.h

	commands/command_gridseg.cpp
	commands/command_gridseg.h
	commands/gridseg_config.cpp
	commands/gridseg_config.h

	commands/command_overlay_segmentation.cpp
	commands/command_overlay_segmentation.h
	commands/overlay_segmentation_config.cpp
	commands/overlay_segmentation_config.h

)

vole_add_command("felzenszwalb" "commands/command_felzenszwalb.h" "vole::CommandFelzenszwalb")
vole_add_command("veksler" "commands/command_veksler.h" "vole::CommandVeksler")
vole_add_command("seg_overlay" "commands/command_overlay_segmentation.h" "vole::CommandOverlaySegmentation")
vole_add_command("gridseg" "commands/command_gridseg.h" "vole::CommandGridSeg")

# module names MUST BE EQUAL to the ones defined in the pyvole_bla.cpp
# BOOST_PYTHON_MODULE(module_name)
vole_add_python_module("_veksler" "commands/pyvole_veksler.cpp")
vole_add_python_module("_felzenszwalb" "commands/pyvole_felzenszwalb.cpp")
vole_add_python_module("_sp" "commands/pyvole_superpixels.cpp")

vole_compile_library(
	superpixel.h
	superpixel_segmentation.cpp
	superpixel_segmentation.h

	normal_image.cpp
	normal_image.h

	felzenszwalb/felzenszwalb_segmentation.cpp
	felzenszwalb/felzenszwalb_segmentation.h

	felzenszwalb/implementation/image.h
	felzenszwalb/implementation/misc.h
	felzenszwalb/implementation/segment-image.h
	felzenszwalb/implementation/filter.h
	felzenszwalb/implementation/segment-graph.h
	felzenszwalb/implementation/disjoint-set.h
	felzenszwalb/implementation/convolve.h
	felzenszwalb/implementation/imconv.h

	patches/patches_segmentation.cpp
	patches/patches_segmentation.h

	veksler/veksler_segmentation.cpp
	veksler/veksler_segmentation.h

	veksler/implementation/maxflow.h
	veksler/implementation/graph.h
	veksler/implementation/superpixels.cpp
	veksler/implementation/superpixels.h
	veksler/implementation/block.h
	veksler/implementation/energy.h

	veksler/implementation/vlib/include/canny.h
	veksler/implementation/vlib/include/draw.h
	veksler/implementation/vlib/include/filter.h
	veksler/implementation/vlib/include/imconv.h
	veksler/implementation/vlib/include/misc.h
	veksler/implementation/vlib/include/point.h
	veksler/implementation/vlib/include/steer.h
	veksler/implementation/vlib/include/chamfer.h
	veksler/implementation/vlib/include/dt.h
	veksler/implementation/vlib/include/grad.h
	veksler/implementation/vlib/include/imutil.h
	veksler/implementation/vlib/include/morph.h
	veksler/implementation/vlib/include/pyramid.h
	veksler/implementation/vlib/include/convolve.h
	veksler/implementation/vlib/include/features.h
	veksler/implementation/vlib/include/image.h
	veksler/implementation/vlib/include/matrix.h
	veksler/implementation/vlib/include/sample.h
)

vole_add_module()
