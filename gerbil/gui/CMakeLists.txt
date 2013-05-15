vole_module_name("gerbil_gui")
vole_module_description("GUI for multispectral images")
vole_module_variable("Gerbil_Viewer")

vole_add_required_dependencies("BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_THREAD" "BOOST_DATE_TIME" "TBB" "OPENCV" "QT" "QT_OPENGL" "OPENGL")
vole_add_required_modules("gerbil_common" "rgb" "seg_graphs")
vole_add_optional_modules("seg_meanshift" "seg_medianshift" "seg_probshift")

vole_compile_library(
	mainwindow
	scaledview
	bandview
	roiview
	commandrunner
	iogui

	dist_view/viewercontainer
	dist_view/multi_img_viewer
	dist_view/viewer_tasks
	dist_view/viewer_bins_tbb
	dist_view/foldingbar
	dist_view/viewport

	model/falsecolor

	tasks/rgbtbb
	tasks/rgbserial
	tasks/normrangetbb
	tasks/normrangecuda

    gerbil_gui_debug
)

vole_moc_library(
	mainwindow.h
	scaledview.h
	bandview.h
	roiview.h
	commandrunner.h

	dist_view/viewercontainer.h
	dist_view/multi_img_viewer.h
	dist_view/viewport.h
	dist_view/foldingbar.h
)

vole_ui_library(
	mainwindow.ui
	dist_view/multi_img_viewer.ui
	dist_view/foldingbar.ui
)

vole_add_resources(
	gerbil.qrc
)

vole_add_executable("qgerbil" "main")

vole_add_module()
