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
	//mw->addDockWidget(Qt::RightDockWidgetArea, illumDock));

	//TODO make this complete
	mw->tabifyDockWidgets(roiDock, rgbDock);

	connect(mw, SIGNAL(requestEnableDocks(bool,TaskType)),
			this, SLOT(enableDocks(bool,TaskType)));
}

void DockController::createDocks()
{
	assert(NULL != mw);
	roiDock = new ROIDock(mw);
	rgbDock = new RgbDock(mw);
}

void DockController::setupDocks()
{
	connect(im, SIGNAL(imageUpdate(representation::t,SharedMultiImgPtr)),
			rgbDock, SLOT(processImageUpdate(representation::t,SharedMultiImgPtr)));

	// TODO: use FalseColorModel
	connect(rgbDock, SIGNAL(rgbRequested()),
			im, SLOT(computeRGB()));

	// TODO: use FalseColorModel
	connect(im, SIGNAL(rgbUpdate(QPixmap)),
			rgbDock, SLOT(updatePixmap(QPixmap)));

	connect(im, SIGNAL(rgbUpdate(QPixmap)),
			this, SLOT(processRGB(QPixmap)));

	// signals for ROI (reset handled in ROIDock)
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

	// TODO create illumDock Dock
	//illumDock->setEnabled((enable || tt == TT_APPLY_ILLUM));

	//TODO
//	usDock->setEnabled(enable && !limitedMode);
	roiDock->setEnabled(enable || tt == TT_SELECT_ROI);
}

void DockController::processRGB(QPixmap rgb)
{
	// TODO FIXME this should be handled the same way as for RGBDock
	GGDBG_CALL();
	roiDock->setPixmap(rgb);

}


