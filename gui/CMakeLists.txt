vole_module_name("gerbil_gui")
vole_module_description("GUI for multispectral images")
vole_module_variable("Gerbil_GUI")

vole_add_required_dependencies("BOOST" "BOOST_PROGRAM_OPTIONS" "TBB" "OPENCV"
    "QT" "QT_OPENGL" "OPENGL" "Threads")
vole_add_required_dependencies("BOOST_THREAD" "BOOST_CHRONO" "BOOST_DATE_TIME")
vole_add_required_modules("rgb" "seg_graphs")
vole_add_optional_modules("seg_meanshift" "seg_medianshift" "seg_probshift" "edge_detect")

vole_compile_library(
	app/gerbilapplication.cpp
	app/gerbilapp_aux.cpp
	app/gerbilio.cpp

	widgets/actionbutton.cpp
	widgets/mainwindow.cpp
	widgets/autohidewidget.cpp
	widgets/autohideview.cpp
	widgets/ahcombobox.cpp
	widgets/scaledview.cpp
	widgets/bandview.cpp
	widgets/roiview.cpp
	widgets/graphsegwidget.cpp
	widgets/modewidget.cpp
	widgets/similaritywidget.cpp

	widgets/sizegripitem/sizegripitem.cpp
	
	controller/controller.cpp
	controller/controller_docks.cpp
	controller/distviewcontroller.cpp
	controller/subscriptions.cpp

	dist_view/distviewcompute.cpp
	dist_view/distviewmodel.cpp
	dist_view/viewer_tasks.cpp
	dist_view/foldingbar.cpp
	dist_view/viewport.cpp
	dist_view/viewport_drawing.cpp
	dist_view/viewport_input.cpp
	dist_view/distviewgui.cpp
	dist_view/distviewbinstbb.cpp

	model/commandrunner.cpp
	model/representation.cpp
	model/imagemodel.cpp
	model/labelingmodel.cpp
	model/falsecolormodel.cpp
	model/falsecolor/falsecoloring.cpp
	model/falsecolor/falsecolormodelpayload.cpp
	model/illuminationmodel.cpp
	model/graphsegmentationmodel.cpp
	model/clusteringmodel.cpp
	model/labels/icontask.cpp

	docks/banddock.cpp
	docks/normdock.cpp
	docks/roidock.cpp
	docks/illumdock.cpp
	docks/falsecolordock.cpp
	docks/clusteringdock.cpp
	docks/labeldock.cpp

	gerbil_gui_debug.cpp

	dialogs/openrecent/recentfile.cpp
	dialogs/openrecent/openrecent.cpp
	dialogs/openrecent/recentfiledelegate.cpp
)

vole_moc_library(
	app/gerbilapplication.h

	widgets/mainwindow.h
	widgets/actionbutton.h
	widgets/autohidewidget.h
	widgets/autohideview.h
	widgets/ahcombobox.h
	widgets/scaledview.h
	widgets/bandview.h
	widgets/roiview.h
	widgets/graphsegwidget.h
	widgets/modewidget.h
	widgets/similaritywidget.h

	widgets/sizegripitem/sizegripitem.h

	controller/controller.h
	controller/distviewcontroller.h

	dist_view/distviewmodel.h
	dist_view/distviewgui.h
	dist_view/foldingbar.h
	dist_view/viewport.h

	docks/banddock.h
	docks/normdock.h
	docks/roidock.h
	docks/illumdock.h
	docks/falsecolordock.h
	docks/clusteringdock.h
	docks/labeldock.h

	model/commandrunner.h
	model/imagemodel.h
	model/labelingmodel.h
	model/labels/icontask.h
	model/falsecolormodel.h
	model/falsecolor/falsecolormodelpayload.h
	model/illuminationmodel.h
	model/graphsegmentationmodel.h
	model/clusteringmodel.h

	dialogs/openrecent/openrecent.h
	dialogs/openrecent/recentfiledelegate.h
)

vole_ui_library(
	widgets/mainwindow.ui
	widgets/graphsegwidget.ui
	widgets/modewidget.ui
	widgets/similaritywidget.ui
	dist_view/distviewgui.ui
	dist_view/foldingbar.ui
	dist_view/viewportcontrol.ui
	docks/roidock.ui
	docks/roidock_buttons.ui
	docks/illumdock.ui
	docks/clusteringdock.ui
	docks/banddock.ui
	docks/normdock.ui
	docks/falsecolordock_sel.ui
	docks/labeldock.ui
	dialogs/openrecent/openrecent.ui
)

vole_add_resources(
	gerbil.qrc
)

vole_add_executable("qgerbil" "app/gerbilapplication.cpp")

vole_add_module()
