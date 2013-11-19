vole_module_name("core")
vole_module_description("Core gerbil functionality")
vole_module_variable("Gerbil_Core")

vole_add_required_dependencies("OPENCV" "TBB")
vole_add_optional_dependencies("BOOST_PROGRAM_OPTIONS" "BOOST_SYSTEM" "BOOST_FILESYSTEM" "QT")
vole_add_optional_dependencies("BOOST_THREAD" "BOOST_DATE_TIME" "BOOST_CHRONO")

vole_compile_library(
	hashes
	vole_config
	qtopencv

	multi_img.h
	multi_img/multi_img
	multi_img/multi_img_io
	multi_img/multi_img_ext
	multi_img/multi_img_io_ext
	multi_img/multi_img_offloaded
	multi_img/multi_img_tbb
	multi_img/illuminant
	multi_img/cieobserver
	background_task/background_task
	background_task/background_task_queue
	background_task/tasks/cuda/gradientcuda
	background_task/tasks/cuda/clampcuda
	background_task/tasks/cuda/datarangecuda
	background_task/tasks/cuda/illuminantcuda
	background_task/tasks/cuda/normrangecuda
	background_task/tasks/tbb/gradienttbb
	background_task/tasks/tbb/clamptbb
	background_task/tasks/tbb/datarangetbb
	background_task/tasks/tbb/illuminanttbb
	background_task/tasks/tbb/rescaletbb
	background_task/tasks/tbb/normrangetbb
	background_task/tasks/tbb/pcatbb
	background_task/tasks/tbb/band2qimagetbb
	background_task/tasks/tbb/rgbqttbb
	background_task/tasks/tbb/bgrtbb
	background_task/tasks/scopeimage

	labeling
	progress_observer
	rectangles
	shared_data
	stopwatch
)

if(QT_FOUND)
vole_moc_library(
	background_task/background_task.h
)
endif()

# fixme python stuff
#vole_add_python_module("_common" "pyvole_common.cpp")

vole_add_module()
