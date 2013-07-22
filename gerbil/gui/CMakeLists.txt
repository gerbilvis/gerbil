vole_module_name("gerbil_gui")
vole_module_description("GUI for multispectral images")
vole_module_variable("Gerbil_Viewer")

vole_add_required_dependencies("BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_THREAD" "BOOST_DATE_TIME" "TBB" "OPENCV" "QT" "QT_OPENGL" "OPENGL")
vole_add_required_modules("gerbil_common" "rgb" "seg_graphs")
vole_add_optional_modules("seg_meanshift" "seg_medianshift" "seg_probshift" "edge_detect")

vole_compile_library(
	autohidewidget
	mainwindow
	scaledview
	bandview
	roiview
	commandrunner
	iogui
	
	controller
	dockcontroller

	dist_view/compute
	dist_view/distviewcontroller
	dist_view/distviewmodel
	dist_view/viewer_tasks
	dist_view/foldingbar
	dist_view/viewport
	dist_view/distviewgui

	model/representation
	model/image
	model/labeling
	model/falsecolor
	model/illumination
	model/graphsegmentation
	model/ussegmentationmodel

	tasks/distviewbinstbb
	tasks/rgbtbb
	tasks/rgbserial
	tasks/normrangetbb
	tasks/normrangecuda

	docks/banddock
	docks/labelingdock
	docks/normdock
	docks/dockwidget
	docks/roidock
	docks/illumdock
	docks/rgbdock
	docks/graphsegwidget
	docks/ussegmentationdock
	docks/labeldock

	gerbil_gui_debug
)

vole_moc_library(
	mainwindow.h
	autohidewidget.h
	scaledview.h
	bandview.h
	roiview.h
	commandrunner.h

	controller.h
	dockcontroller.h

	dist_view/distviewmodel.h
	dist_view/distviewcontroller.h
	dist_view/distviewgui.h
	dist_view/foldingbar.h
	dist_view/viewport.h

	docks/banddock.h
	docks/labelingdock.h
	docks/normdock.h
	docks/dockwidget.h
	docks/roidock.h
	docks/illumdock.h
	docks/rgbdock.h
	docks/graphsegwidget.h
	docks/ussegmentationdock.h
	docks/labeldock.h

	model/image.h
	model/labeling.h
	model/falsecolor.h
	model/illumination.h
	model/graphsegmentation.h
	model/ussegmentationmodel.h
)

vole_ui_library(
	mainwindow.ui
	dist_view/distviewgui.ui
	dist_view/foldingbar.ui
	dist_view/viewportcontrol.ui
	docks/roidock.ui
	docks/illumdock.ui
	docks/graphsegwidget.ui
	docks/ussegmentationdock.ui
	docks/banddock.ui
	docks/labelingdock.ui
	docks/normdock.ui
	docks/rgbdock.ui
	docks/labeldock.ui
)

vole_add_resources(
	gerbil.qrc
)

vole_add_executable("qgerbil" "main")

vole_add_module()
