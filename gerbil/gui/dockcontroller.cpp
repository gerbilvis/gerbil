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
	mw->addDockWidget(Qt::RightDockWidgetArea, graphSegDock);
#ifdef WITH_SEG_MEANSHIFT
	mw->addDockWidget(Qt::RightDockWidgetArea, usSegDock);
#endif
	mw->addDockWidget(Qt::RightDockWidgetArea, rgbDock);
	mw->addDockWidget(Qt::RightDockWidgetArea, roiDock);
	mw->addDockWidget(Qt::RightDockWidgetArea, illumDock);

	im->computeFullRgb();

	//TODO make this complete
	mw->tabifyDockWidgets(roiDock, rgbDock, illumDock, graphSegDock, usSegDock);

	connect(chief, SIGNAL(requestEnableDocks(bool,TaskType)),
			this, SLOT(enableDocks(bool,TaskType)));
}

void DockController::createDocks()
{
	assert(NULL != mw);
	roiDock = new ROIDock(mw);
	illumDock = new IllumDock(mw);
	rgbDock = new RgbDock(mw);
	graphSegDock = new GraphSegmentationDock(mw);
#ifdef WITH_SEG_MEANSHIFT
	usSegDock = new UsSegmentationDock(mw);
#endif
}

void DockController::setupDocks()
{
	/* RGB Dock */
	connect(im, SIGNAL(imageUpdate(representation::t,SharedMultiImgPtr)),
			rgbDock, SLOT(processImageUpdate(representation::t,SharedMultiImgPtr)));

	connect(rgbDock, SIGNAL(rgbRequested(coloring)),
			fm, SLOT(computeBackground(coloring)));

	connect(fm, SIGNAL(calculationComplete(coloring, QPixmap)),
			rgbDock, SLOT(updatePixmap(coloring, QPixmap)));

	/* ROI Dock */
	// signals for ROI (reset handled in ROIDock)
	connect(im, SIGNAL(fullRgbUpdate(QPixmap)),
			roiDock, SLOT(updatePixmap(QPixmap)));

	connect(roiDock, SIGNAL(roiRequested(const cv::Rect&)),
			chief, SLOT(spawnROI(const cv::Rect&)));

	/* Illumination Dock */
	connect(illumDock, SIGNAL(applyIllum()),
			illumm, SLOT(applyIllum()));
	connect(illumDock, SIGNAL(illum1Selected(int)),
			illumm, SLOT(updateIllum1(int))); //FIXME slot name
	connect(illumDock, SIGNAL(illum2Selected(int)),
			illumm, SLOT(updateIllum2(int)));
	connect(illumDock, SIGNAL(showIlluminationCurveChanged(bool)),
			illumm, SLOT(setIlluminationCurveShown(bool)));

	// TODO: connections between illumDock and viewer container
	connect(illumDock, SIGNAL(showIlluminationCurveChanged(bool)),
			mw->getViewerContainer(), SLOT(showIlluminationCurve(bool)));



	/* Graph Segmentation Dock */

	// TODO more
	connect(mw, SIGNAL(graphSegDockVisibleRequested(bool)),
			graphSegDock, SLOT(setVisible(bool)));
	graphSegDock->setVisible(false); // start hidden

	/* Unsupervised Segmentation Dock */
#ifdef WITH_SEG_MEANSHIFT
	int nbands = im->getSize();
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
	usSegDock->setEnabled(enable && !im->isLimitedMode());
#endif
	roiDock->setEnabled(enable || tt == TT_SELECT_ROI);

	graphSegDock->setEnabled(enable);
}
