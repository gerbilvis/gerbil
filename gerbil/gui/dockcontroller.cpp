#include "dockcontroller.h"

#include "controller.h"

#include "mainwindow.h"

#include "docks/rgbdock.h"
#include "docks/roidock.h"
#include "docks/illumdock.h"
#include "docks/graphsegmentationdock.h"
#include "docks/ussegmentationdock.h"

#include "model/ussegmentationmodel.h"

DockController::DockController(Controller *chief) :
	QObject(chief), chief(chief)
{
}

void DockController::init()
{
	createDocks();
	setupDocks();

	// FIXME: Put graphSegDock below band view dock.
	// Can't do this here until *all* docks are added here instead of being created
	// in the mainwindow ui file. (well we could drag programmatically -> arghl).
	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, graphSegDock);
#ifdef WITH_SEG_MEANSHIFT
	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, usSegDock);
#endif
	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, rgbDock);
	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, roiDock);
	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, illumDock);

	chief->imageModel()->computeFullRgb();

	//TODO make this complete
	chief->mainWindow()->tabifyDockWidgets(
				roiDock, rgbDock, illumDock, graphSegDock, usSegDock);

	connect(chief, SIGNAL(requestEnableDocks(bool,TaskType)),
			this, SLOT(enableDocks(bool,TaskType)));
}

void DockController::createDocks()
{
	assert(NULL != chief->mainWindow());
	roiDock = new ROIDock(chief->mainWindow());
	illumDock = new IllumDock(chief->mainWindow());
	rgbDock = new RgbDock(chief->mainWindow());
	graphSegDock = new GraphSegmentationDock(chief->mainWindow());
#ifdef WITH_SEG_MEANSHIFT
	usSegDock = new UsSegmentationDock(chief->mainWindow());
#endif
}

void DockController::setupDocks()
{
	/* RGB Dock */
	connect(chief->imageModel(), SIGNAL(imageUpdate(representation::t,SharedMultiImgPtr)),
			rgbDock, SLOT(processImageUpdate(representation::t,SharedMultiImgPtr)));

	connect(rgbDock, SIGNAL(rgbRequested(coloring)),
			chief->falseColorModel(), SLOT(computeBackground(coloring)));

	connect(chief->falseColorModel(), SIGNAL(calculationComplete(coloring, QPixmap)),
			rgbDock, SLOT(updatePixmap(coloring, QPixmap)));

	/* ROI Dock */
	// signals for ROI (reset handled in ROIDock)
	connect(chief->imageModel(), SIGNAL(fullRgbUpdate(QPixmap)),
			roiDock, SLOT(updatePixmap(QPixmap)));

	connect(roiDock, SIGNAL(roiRequested(const cv::Rect&)),
			chief, SLOT(spawnROI(const cv::Rect&)));

	/* Illumination Dock */
	connect(illumDock, SIGNAL(applyIllum()),
			chief->illumModel(), SLOT(applyIllum()));
	connect(illumDock, SIGNAL(illum1Selected(int)),
			chief->illumModel(), SLOT(updateIllum1(int))); //FIXME slot name
	connect(illumDock, SIGNAL(illum2Selected(int)),
			chief->illumModel(), SLOT(updateIllum2(int)));
	connect(illumDock, SIGNAL(showIlluminationCurveChanged(bool)),
			chief->illumModel(), SLOT(setIlluminationCurveShown(bool)));

	// TODO: connections between illumDock and viewer container
	connect(illumDock, SIGNAL(showIlluminationCurveChanged(bool)),
			chief->mainWindow()->getViewerContainer(), SLOT(showIlluminationCurve(bool)));



	/* Graph Segmentation Dock */

	// TODO more
	connect(chief->mainWindow(), SIGNAL(graphSegDockVisibleRequested(bool)),
			graphSegDock, SLOT(setVisible(bool)));
	graphSegDock->setVisible(false); // start hidden

	/* Unsupervised Segmentation Dock */
#ifdef WITH_SEG_MEANSHIFT
	UsSegmentationModel const*um = chief->usSegmentationModel();
	int nbands = chief->imageModel()->getSize();
	usSegDock->setNumBands(nbands);
	connect(chief, SIGNAL(nSpectralBandsChanged(int)),
			usSegDock, SLOT(setNumBands(int)));
	connect(um, SIGNAL(progressChanged(int)),
			usSegDock, SLOT(updateProgress(int)));
	connect(um, SIGNAL(segmentationCompleted()),
			usSegDock, SLOT(processSegmentationCompleted()));
	connect(usSegDock, SIGNAL(segmentationRequested(vole::Command*,int,bool)),
			um, SLOT(startSegmentation(vole::Command*,int,bool)));
	connect(usSegDock, SIGNAL(cancelSegmentationRequested()),
			um, SLOT(cancel()));
	// FIXME: 2013-06-17 altmann
	// If enabled, gerbil crashes. I am not familiar to the labeling stuff.
	// Probably need to connect to different slot in LabelingModel or the computed
	// labeling is inconsistent with the current state in LabelingModel.
	// connect(um, SIGNAL(setLabelsRequested(cv::Mat1s)),
	//			lm, SLOT(setLabels(cv::Mat1s)));

	// FIXME hide for release?
	//usSegDock->hide();
#endif /* WITH_SEG_MEANSHIFT */

}


void DockController::enableDocks(bool enable, TaskType tt)
{
	//TODO
	//	labelDock->setEnabled(enable);
	rgbDock->setEnabled(enable);

	// TODO limitedMode - availabe from Controller?
	//illumDock->setEnabled((enable || tt == TT_APPLY_ILLUM) && !im->isLimitedMode());
	illumDock->setEnabled((enable || tt == TT_APPLY_ILLUM));

#ifdef WITH_SEG_MEANSHIFT
	usSegDock->setEnabled(enable && !chief->imageModel()->isLimitedMode());
#endif
	roiDock->setEnabled(enable || tt == TT_SELECT_ROI);

	graphSegDock->setEnabled(enable);
}
