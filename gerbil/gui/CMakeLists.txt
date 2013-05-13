vole_module_name("gerbil_gui")
vole_module_description("GUI for multispectral images")
vole_module_variable("Gerbil_Viewer")

vole_add_required_dependencies("BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_THREAD" "BOOST_DATE_TIME" "TBB" "OPENCV" "QT" "QT_OPENGL" "OPENGL")
vole_add_required_modules("gerbil_common" "seg_graphs")
vole_add_optional_modules("seg_meanshift" "seg_medianshift" "seg_probshift")

vole_compile_library(
	multi_img_viewer
	mainwindow
	viewercontainer
	viewport
	scaledview
	bandview
	roiview
	commandrunner
	foldingbar
	iogui
	model/falsecolor
	tasks/rgbtbb
	tasks/rgbserial
	tasks/normrangetbb
	tasks/normrangecuda
	multi_img_viewer_detail/viewer_tasks
	multi_img_viewer_detail/viewer_bins_tbb
    gerbil_gui_debug
)

vole_moc_library(
	multi_img_viewer.h
	mainwindow.h
	viewercontainer.h
	viewport.h
	scaledview.h
	bandview.h
	roiview.h
	commandrunner.h
	foldingbar.h
)

vole_ui_library(
	multi_img_viewer.ui
	mainwindow.ui
	foldingbar.ui
)

vole_add_resources(
	gerbil.qrc
)

vole_add_executable("qgerbil" "main")

vole_add_module()
