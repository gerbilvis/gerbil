#include "dockcontroller.h"

#include "controller.h"

#include "mainwindow.h"

#include "docks/rgbdock.h"
#include "docks/roidock.h"
#include "docks/illumdock.h"


DockController::DockController(Controller *chief) :
	QObject(chief), chief(chief)
{
}

void DockController::init()
{
	createDocks();
	setupDocks();

	mw->addDockWidget(Qt::RightDockWidgetArea, roiDock);
	mw->addDockWidget(Qt::RightDockWidgetArea, rgbDock);
	mw->addDockWidget(Qt::RightDockWidgetArea, illumDock);

	im->computeFullRgb();

	//TODO make this complete
	mw->tabifyDockWidgets(roiDock, rgbDock, illumDock);

	connect(mw, SIGNAL(requestEnableDocks(bool,TaskType)),
			this, SLOT(enableDocks(bool,TaskType)));
}

void DockController::createDocks()
{
	assert(NULL != mw);
	roiDock = new ROIDock(mw);
	illumDock = new IllumDock(mw);
	rgbDock = new RgbDock(mw);
}

void DockController::setupDocks()
{
	connect(im, SIGNAL(imageUpdate(representation::t,SharedMultiImgPtr)),
			rgbDock, SLOT(processImageUpdate(representation::t,SharedMultiImgPtr)));

	// siganls for the illumDock
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

	// signals for the rgbDock
	connect(rgbDock, SIGNAL(rgbRequested(coloring)),
			fm, SLOT(computeBackground(coloring)));

	connect(fm, SIGNAL(calculationComplete(coloring, QPixmap)),
			rgbDock, SLOT(updatePixmap(coloring, QPixmap)));


	// signals for ROI (reset handled in ROIDock)
	connect(im, SIGNAL(fullRgbUpdate(QPixmap)),
			roiDock, SLOT(updatePixmap(QPixmap)));

	connect(roiDock, SIGNAL(roiRequested(const cv::Rect&)),
			chief, SLOT(spawnROI(const cv::Rect&)));
}


void DockController::enableDocks(bool enable, TaskType tt)
{
	//TODO
//	labelDock->setEnabled(enable);
	rgbDock->setEnabled(enable);

	// TODO limitedMode - availabe from Controller?
	//illumDock->setEnabled((enable || tt == TT_APPLY_ILLUM) && !limitedMode);
	illumDock->setEnabled((enable || tt == TT_APPLY_ILLUM));

	//TODO
//	usDock->setEnabled(enable && !limitedMode);
	roiDock->setEnabled(enable || tt == TT_SELECT_ROI);
}

// TODO: remove
void DockController::processRGB(QPixmap rgb)
{
	// TODO FIXME this should be handled the same way as for RGBDock
	GGDBG_CALL();
}


