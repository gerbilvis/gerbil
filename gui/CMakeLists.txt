vole_module_name("gerbil_gui")
vole_module_description("GUI for multispectral images")
vole_module_variable("Gerbil_GUI")

vole_add_required_dependencies("BOOST" "BOOST_PROGRAM_OPTIONS" "TBB" "OPENCV" "QT" "QT_OPENGL" "OPENGL")
vole_add_required_dependencies("BOOST_THREAD" "BOOST_CHRONO" "BOOST_DATE_TIME")
vole_add_required_modules("rgb" "seg_graphs")
vole_add_optional_modules("seg_meanshift" "seg_medianshift" "seg_probshift" "edge_detect")

vole_compile_library(
	main
	iogui
	commandrunner

	widgets/mainwindow
	widgets/autohidewidget
	widgets/autohideview
	widgets/ahcombobox
	widgets/scaledview
	widgets/bandview
	widgets/roiview
	widgets/graphsegwidget

	widgets/sizegripitem/sizegripitem
	
	controller/controller
	controller/dockcontroller
	controller/distviewcontroller

	dist_view/compute
	dist_view/distviewmodel
	dist_view/viewer_tasks
	dist_view/foldingbar
	dist_view/viewport
	dist_view/viewport_drawing
	dist_view/viewport_input
	dist_view/distviewgui
	dist_view/distviewbinstbb

	model/representation
	model/imagemodel
	model/labelingmodel
	model/falsecolormodel
	model/falsecolor/falsecoloring
	model/falsecolor/falsecolormodelpayload
	model/illuminationmodel
	model/graphsegmentationmodel
	model/clusteringmodel
	model/labels/icontask


	docks/banddock
	docks/labelingdock
	docks/normdock
	docks/roidock
	docks/illumdock
	docks/falsecolordock
	docks/clusteringdock
	docks/labeldock

	gerbil_gui_debug
)

vole_moc_library(
	commandrunner.h

	widgets/mainwindow.h
	widgets/autohidewidget.h
	widgets/autohideview.h
	widgets/ahcombobox.h
	widgets/scaledview.h
	widgets/bandview.h
	widgets/roiview.h
	widgets/graphsegwidget.h

	widgets/sizegripitem/sizegripitem.h

	controller/controller.h
	controller/dockcontroller.h
	controller/distviewcontroller.h

	dist_view/distviewmodel.h
	dist_view/distviewgui.h
	dist_view/foldingbar.h
	dist_view/viewport.h

	docks/banddock.h
	docks/labelingdock.h
	docks/normdock.h
	docks/roidock.h
	docks/illumdock.h
	docks/falsecolordock.h
	docks/clusteringdock.h
	docks/labeldock.h

	model/imagemodel.h
	model/labelingmodel.h
	model/labels/icontask.h
	model/falsecolormodel.h
	model/falsecolor/falsecolormodelpayload.h
	model/illuminationmodel.h
	model/graphsegmentationmodel.h
	model/clusteringmodel.h
)

vole_ui_library(
	widgets/mainwindow.ui
	widgets/graphsegwidget.ui
	dist_view/distviewgui.ui
	dist_view/foldingbar.ui
	dist_view/viewportcontrol.ui
	docks/roidock.ui
	docks/roidock_buttons.ui
	docks/illumdock.ui
	docks/clusteringdock.ui
	docks/banddock.ui
	docks/labelingdock.ui
	docks/normdock.ui
	docks/falsecolordock_sel.ui
	docks/labeldock.ui
)

vole_add_resources(
	gerbil.qrc
)

vole_add_executable("qgerbil" "main")

vole_add_module()
