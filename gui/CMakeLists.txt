vole_module_name("gerbil_gui")
vole_module_description("GUI for multispectral images")
vole_module_variable("Gerbil_Viewer")

vole_add_required_dependencies("BOOST" "BOOST_PROGRAM_OPTIONS" "OPENCV" "QT" "QT_OPENGL")
vole_add_required_modules("gerbil_common" "seg_graphs")
vole_add_optional_modules("seg_meanshift" "seg_medianshift" "seg_probshift")

vole_compile_library(
	multi_img_viewer
	viewerwindow
	viewport
	scaledview
	bandview
	roiview
	commandrunner
	foldingbar
)

vole_moc_library(
	multi_img_viewer.h
	viewerwindow.h
	viewport.h
	scaledview.h
	bandview.h
	roiview.h
	commandrunner.h
	foldingbar.h
)

vole_ui_library(
	multi_img_viewer.ui
	viewerwindow.ui
	foldingbar.ui
)

vole_add_resources(
	gerbil.qrc
)

vole_add_executable("qgerbil" "main")

vole_add_module()
