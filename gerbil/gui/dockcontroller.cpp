#include "dockcontroller.h"

#include "controller.h"

#include "mainwindow.h"

#include "docks/banddock.h"
#include "docks/labelingdock.h"
#include "docks/normdock.h"
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

	//TODO: re-implement NormDock

	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, bandDock);
	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, graphSegDock);
	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, labelingDock);
	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, normDock);
#ifdef WITH_SEG_MEANSHIFT
	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, usSegDock);
	usSegDock->setVisible(false);
#endif
	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, rgbDock);
	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, roiDock);
	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, illumDock);

	// dock arrangement
	chief->mainWindow()->tabifyDockWidget(roiDock, rgbDock);
#ifdef WITH_SEG_MEANSHIFT
	chief->mainWindow()->tabifyDockWidget(roiDock, usSegDock);
#endif
	roiDock->raise();

	chief->mainWindow()->tabifyDockWidget(labelingDock, illumDock);
	chief->mainWindow()->tabifyDockWidget(labelingDock, normDock);
	chief->mainWindow()->tabifyDockWidget(labelingDock, graphSegDock);

	chief->imageModel()->computeFullRgb();

	connect(chief, SIGNAL(requestEnableDocks(bool,TaskType)),
			this, SLOT(enableDocks(bool,TaskType)));
}

void DockController::createDocks()
{
	assert(NULL != chief->mainWindow());
	bandDock = new BandDock(chief->mainWindow());
	labelingDock = new LabelingDock(chief->mainWindow());
	normDock = new NormDock(chief->mainWindow());
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

	/* im -> others */
	connect(chief->imageModel(), SIGNAL(bandUpdate(QPixmap, QString)),
			bandDock, SLOT(changeBand(QPixmap, QString)));

	/* Band Dock */
	connect(chief->labelingModel(), SIGNAL(partialLabelUpdate(const cv::Mat1s&,const cv::Mat1b&)),
			bandDock, SLOT(processLabelingChange(cv::Mat1s,cv::Mat1b)));
	connect(chief->labelingModel(), SIGNAL(newLabeling(cv::Mat1s,QVector<QColor>,bool)),
			bandDock, SLOT(processLabelingChange(cv::Mat1s,QVector<QColor>,bool)));

	connect(bandDock->bandView(), SIGNAL(alteredLabels(cv::Mat1s,cv::Mat1b)),
			chief->labelingModel(), SLOT(alterPixels(cv::Mat1s,cv::Mat1b)));
	connect(bandDock->bandView(), SIGNAL(newLabeling(cv::Mat1s)),
			chief->labelingModel(), SLOT(setLabels(cv::Mat1s)));

	connect(bandDock->bandView(), SIGNAL(killHover()),
			chief->mainWindow()->getViewerContainer(), SIGNAL(viewportsKillHover()));
	connect(bandDock->bandView(), SIGNAL(pixelOverlay(int,int)),
			chief->mainWindow()->getViewerContainer(), SIGNAL(viewersOverlay(int,int)));
	connect(bandDock->bandView(), SIGNAL(newSingleLabel(short)),
			chief->mainWindow()->getViewerContainer(), SIGNAL(viewersHighlight(short)));
	connect(bandDock, SIGNAL(currentLabelChanged(int)),
			chief->mainWindow(), SLOT(setCurrentLabel(int)));
	// alterLabel(short) -> clear label
	connect(bandDock, SIGNAL(clearLabelRequested(short)),
			chief->labelingModel(), SLOT(alterLabel(short)));
	connect(bandDock, SIGNAL(newLabelRequested()),
			chief->labelingModel(), SLOT(addLabel()));
	connect(bandDock, SIGNAL(graphSegModeToggled(bool)),
			graphSegDock, SLOT(setVisible(bool)));
	connect(bandDock, SIGNAL(graphSegModeToggled(bool)),
			this, SLOT(raiseGraphSegDock(bool)));

	// GraphSegModel -> BandDock
	connect(chief->graphSegmentationModel(), SIGNAL(seedingDone()),
			bandDock, SLOT(processSeedingDone()));

	connect(chief->mainWindow()->getViewerContainer(), SIGNAL(drawOverlay(const cv::Mat1b&)),
		bandDock->bandView(), SLOT(drawOverlay(const cv::Mat1b&)));
	connect(chief->mainWindow(), SIGNAL(ignoreLabelsRequested(bool)),
			bandDock->bandView(), SLOT(toggleShowLabels(bool)));
	connect(chief->mainWindow(), SIGNAL(singleLabelRequested(bool)),
			bandDock->bandView(), SLOT(toggleSingleLabel(bool)));

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

	/* Labeling Dock */
	connect(labelingDock, SIGNAL(requestLoadLabeling()),
			chief->labelingModel(), SLOT(loadLabeling()));
	connect(labelingDock, SIGNAL(requestSaveLabeling()),
			chief->labelingModel(), SLOT(saveLabeling()));
	// TODO
	//connect(labelingDock, SIGNAL(requestLoadSeeds()),
	//		...);

	/* Illumination Dock */
	connect(illumDock, SIGNAL(applyIllum()),
			chief->illumModel(), SLOT(applyIllum()));
	connect(illumDock, SIGNAL(illum1Selected(int)),
			chief->illumModel(), SLOT(updateIllum1(int))); //FIXME slot name
	connect(illumDock, SIGNAL(illum2Selected(int)),
			chief->illumModel(), SLOT(updateIllum2(int)));
	connect(illumDock, SIGNAL(showIlluminationCurveChanged(bool)),
			chief->illumModel(), SLOT(setIlluminationCurveShown(bool)));

	// connections between illumDock and viewer container
	connect(illumDock, SIGNAL(showIlluminationCurveChanged(bool)),
			chief->mainWindow()->getViewerContainer(), SLOT(showIlluminationCurve(bool)));



	/* Graph Segmentation Dock */
	chief->graphSegmentationModel()->setSeedMap(
		bandDock->bandView()->getSeedMap());
	connect(bandDock, SIGNAL(currentLabelChanged(int)),
			chief->graphSegmentationModel(), SLOT(setCurLabel(int)));

	connect(chief->mainWindow(), SIGNAL(graphSegDockVisibleRequested(bool)),
			graphSegDock, SLOT(setVisible(bool)));
	connect(graphSegDock,
			SIGNAL(requestGraphseg(representation::t,vole::GraphSegConfig)),
			chief->graphSegmentationModel(),
			SLOT(runGraphseg(representation::t,vole::GraphSegConfig)));
	connect(graphSegDock, SIGNAL(requestGraphsegCurBand(vole::GraphSegConfig)),
			chief->graphSegmentationModel(),
			SLOT(runGraphsegBand(vole::GraphSegConfig)));
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

	/* Normalization Dock */
	connect(chief->imageModel(),
			SIGNAL(dataRangeUdpate(representation::t,ImageDataRange)),
			normDock,
			SLOT(setNormRange(representation::t,ImageDataRange)));
	connect(normDock,
			SIGNAL(computeDataRangeRequested(representation::t)),
			chief->imageModel(),
			SLOT(computeDataRange(representation::t)));
	connect(normDock,
			SIGNAL(normalizationParametersChanged(
					   representation::t,MultiImg::NormMode,ImageDataRange)),
			chief->imageModel(),
			SLOT(setNormalizationParameters(
					 representation::t,MultiImg::NormMode,ImageDataRange)));
	connect(normDock, SIGNAL(applyNormalizationRequested()),
			chief,
			SLOT(invalidateROI()));

}

void DockController::raiseGraphSegDock(bool enable)
{
	if(enable) {
		//GGDBGM("raising graph seg dock"<<endl);
		graphSegDock->setVisible(true);
		graphSegDock->raise();
	}
}


void DockController::enableDocks(bool enable, TaskType tt)
{
	bandDock->setEnabled(enable);
	bandDock->bandView()->setEnabled(enable);

	if (tt == TT_SELECT_ROI && (!enable)) {
		/* TODO: check if this is enough to make sure no label changes
		 * happen during ROI recomputation */
		bandDock->bandView()->commitLabelChanges();
	}

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
