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
			this, SLOT(processNewImageData(representation::t,SharedMultiImgPtr)));

	connect(this, SIGNAL(rgbRequested()),
			im, SLOT(computeRGB()));

	connect(im, SIGNAL(rgbUpdate(QPixmap)),
			this, SLOT(processRGB(QPixmap)));

	connect(rgbDock, SIGNAL(visibilityChanged(bool)),
			this, SLOT(setRgbVisible(bool)));

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

void DockController::processNewImageData(representation::t type, SharedMultiImgPtr image)
{
	GGDBG_CALL();
	rgbImageValid=false;
	if(type==representation::IMG && rgbVisible) {
		emit rgbRequested();
	}
}

void DockController::processRGB(QPixmap rgb)
{
	GGDBG_CALL();
	rgbDock->updatePixmap(rgb);
	roiDock->setPixmap(rgb);
	rgbImageValid=true;

	/* TODO: in the future, rgbView is independent from this and feeds from
	 * falsecolor model. We could think about data-sharing between image model
	 * and falsecolor model for the CMF part.
	 */
	/*TODO2: move this to apply roi! or sth. like that
	QPixmap rgbroi = rgb.copy(roi.x, roi.y, roi.width, roi.height);
	rgbView->setPixmap(rgbroi);
	rgbView->update();*/
}

void DockController::setRgbVisible(bool visible)
{
	rgbVisible = visible;
	GGDBGM(format("visible=%1% valid=%2%\n") %visible %rgbImageValid );
	if(rgbVisible && !rgbImageValid) {
		emit rgbRequested();
	}
}
