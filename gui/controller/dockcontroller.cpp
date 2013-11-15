#include "dockcontroller.h"

#include "controller.h"

#include "widgets/mainwindow.h"
#include "widgets/graphsegwidget.h"

#include "docks/banddock.h"
#include "widgets/bandview.h"
#include "docks/labelingdock.h"
#include "docks/normdock.h"
#include "docks/falsecolordock.h"
#include "docks/roidock.h"
#include "docks/illumdock.h"
#include "docks/clusteringdock.h"
#include "docks/labeldock.h"

#include "model/labelingmodel.h"
#include "model/falsecolormodel.h"
#include "model/graphsegmentationmodel.h"
#include "model/clusteringmodel.h"

#include "gerbil_gui_debug.h"

DockController::DockController(Controller *chief) :
	QObject(chief), chief(chief)
{
}

void DockController::init()
{
	createDocks();
	setupDocks();

	/// left side
	chief->mainWindow()->addDockWidget(Qt::LeftDockWidgetArea, roiDock);
	chief->mainWindow()->addDockWidget(Qt::LeftDockWidgetArea, illumDock);
	chief->mainWindow()->addDockWidget(Qt::LeftDockWidgetArea, normDock);
	chief->mainWindow()->tabifyDockWidget(illumDock, normDock);
#ifdef WITH_SEG_MEANSHIFT
	chief->mainWindow()->addDockWidget(Qt::LeftDockWidgetArea, clusteringDock);
#endif
	chief->mainWindow()->addDockWidget(Qt::LeftDockWidgetArea, labelingDock);

	/// right side
	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, bandDock);
	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, labelDock);
	chief->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, falseColorDock);

	// dock arrangement
//	chief->mainWindow()->tabifyDockWidget(roiDock, falseColorDock);
#ifdef WITH_SEG_MEANSHIFT
//	chief->mainWindow()->tabifyDockWidget(roiDock, clusteringDock);
#endif
	roiDock->raise();

	chief->imageModel()->computeFullRgb();
}

void DockController::createDocks()
{
	assert(NULL != chief->mainWindow());

	bandDock = new BandDock(chief->imageModel()->getFullImageRect(),
							chief->mainWindow());
	labelingDock = new LabelingDock(chief->mainWindow());
	normDock = new NormDock(chief->mainWindow());
	roiDock = new RoiDock(chief->mainWindow());
	illumDock = new IllumDock(chief->mainWindow());
	falseColorDock = new FalseColorDock(chief->mainWindow());
#ifdef WITH_SEG_MEANSHIFT
	clusteringDock = new ClusteringDock(chief->mainWindow());
#endif
	labelDock = new LabelDock(chief->mainWindow());
}

void DockController::setupDocks()
{

	/* im -> others */
	connect(chief->imageModel(),
			SIGNAL(bandUpdate(representation::t, int, QPixmap, QString)),
			bandDock,
			SLOT(changeBand(representation::t, int, QPixmap, QString)));

	/* Band Dock */
	connect(chief->labelingModel(), SIGNAL(partialLabelUpdate(const cv::Mat1s&,const cv::Mat1b&)),
			bandDock, SLOT(processLabelingChange(cv::Mat1s,cv::Mat1b)));
	connect(chief->imageModel(), SIGNAL(imageUpdate(representation::t,SharedMultiImgPtr)),
			bandDock, SLOT(processImageUpdate(representation::t)));
	connect(chief->labelingModel(), SIGNAL(newLabeling(cv::Mat1s,QVector<QColor>,bool)),
			bandDock, SLOT(processLabelingChange(cv::Mat1s,QVector<QColor>,bool)));

	connect(bandDock, SIGNAL(bandRequested(representation::t, int)),
			chief->imageModel(), SLOT(computeBand(representation::t, int)));
	connect(bandDock->bandView(), SIGNAL(alteredLabels(cv::Mat1s,cv::Mat1b)),
			chief->labelingModel(), SLOT(alterPixels(cv::Mat1s,cv::Mat1b)));
	connect(bandDock->bandView(), SIGNAL(newLabeling(cv::Mat1s)),
			chief->labelingModel(), SLOT(setLabels(cv::Mat1s)));

	connect(bandDock->bandView(), SIGNAL(pixelOverlay(int,int)),
			chief, SIGNAL(requestPixelOverlay(int,int)));
	connect(bandDock->bandView(), SIGNAL(singleLabelSelected(int)),
			chief, SIGNAL(singleLabelSelected(int)));
	connect(bandDock, SIGNAL(currentLabelChanged(int)),
			chief, SIGNAL(currentLabelChanged(int)));
	// alterLabel(short) -> clear label
	connect(bandDock, SIGNAL(clearLabelRequested(short)),
			chief->labelingModel(), SLOT(alterLabel(short)));
	connect(bandDock, SIGNAL(newLabelRequested()),
			chief->labelingModel(), SLOT(addLabel()));

	/* Graph Segmentation Widget */
	// DockController adds missing information and resends the signal
	connect(bandDock->graphSegWidget(),
			SIGNAL(requestGraphseg(representation::t,
								   vole::GraphSegConfig,bool)),
			this,
			SLOT(requestGraphseg(representation::t,
							 vole::GraphSegConfig,bool)));
	connect(this,
			SIGNAL(requestGraphseg(representation::t,cv::Mat1s,
								   vole::GraphSegConfig,bool)),
			chief->graphSegmentationModel(),
			SLOT(runGraphseg(representation::t,cv::Mat1s,
							 vole::GraphSegConfig,bool)));
	connect(bandDock->graphSegWidget(),
			SIGNAL(requestGraphsegCurBand(const vole::GraphSegConfig &,bool)),
			this,
			SLOT(requestGraphsegCurBand(const vole::GraphSegConfig &,bool)));
	connect(this,
			SIGNAL(requestGraphsegBand(representation::t,int,cv::Mat1s,
									   const vole::GraphSegConfig &,bool)),
			chief->graphSegmentationModel(),
			SLOT(runGraphsegBand(representation::t,int,cv::Mat1s,
								 const vole::GraphSegConfig &,bool)));

	// GraphSegModel -> BandDock
	connect(bandDock, SIGNAL(currentLabelChanged(int)),
			chief->graphSegmentationModel(), SLOT(setCurLabel(int)));
	connect(chief->graphSegmentationModel(), SIGNAL(seedingDone()),
			bandDock->graphSegWidget(), SLOT(processSeedingDone()));

	connect(chief, SIGNAL(requestOverlay(const cv::Mat1b&)),
			bandDock->bandView(), SLOT(drawOverlay(const cv::Mat1b&)));

	connect(chief, SIGNAL(toggleIgnoreLabels(bool)),
			bandDock->bandView(), SLOT(toggleShowLabels(bool)));
	connect(chief, SIGNAL(toggleSingleLabel(bool)),
			bandDock->bandView(), SLOT(toggleSingleLabel(bool)));

	/* FalseColor Dock */
	connect(falseColorDock, SIGNAL(falseColoringRequested(FalseColoring::Type,bool)),
			chief->falseColorModel(), SLOT(requestColoring(FalseColoring::Type,bool)));
	connect(falseColorDock, SIGNAL(cancelComputationRequested(FalseColoring::Type)),
			chief->falseColorModel(), SLOT(cancelComputation(FalseColoring::Type)));

	connect(chief->falseColorModel(), SIGNAL(coloringOutOfDate(FalseColoring::Type)),
			falseColorDock, SLOT(processColoringOutOfDate(FalseColoring::Type)));
	connect(chief->falseColorModel(), SIGNAL(progressChanged(FalseColoring::Type,int)),
			falseColorDock, SLOT(processCalculationProgressChanged(FalseColoring::Type,int)));
	connect(chief->falseColorModel(), SIGNAL(coloringComputed(FalseColoring::Type,QPixmap)),
			falseColorDock, SLOT(processColoringComputed(FalseColoring::Type,QPixmap)));
	connect(chief->falseColorModel(), SIGNAL(computationCancelled(FalseColoring::Type)),
			falseColorDock, SLOT(processComputationCancelled(FalseColoring::Type)));

	// needed for ROI dock, clustering dock
	int nbands = chief->imageModel()->getNumBandsFull();

	/* ROI Dock */
	roiDock->setMaxBands(nbands);
	// model to dock (reset handled in RoiDock)
	connect(chief->imageModel(), SIGNAL(fullRgbUpdate(QPixmap)),
			roiDock, SLOT(updatePixmap(QPixmap)));

	connect(chief->imageModel(), SIGNAL(roiRectChanged(cv::Rect)),
			roiDock, SLOT(setRoi(cv::Rect)));

	// dock to controller
	connect(roiDock, SIGNAL(roiRequested(const cv::Rect&)),
			chief, SLOT(spawnROI(const cv::Rect&)));
	connect(roiDock, SIGNAL(specRescaleRequested(int)),
			chief, SLOT(rescaleSpectrum(int)));


	/* Labeling Dock */
	connect(labelingDock, SIGNAL(requestLoadLabeling()),
			chief->labelingModel(), SLOT(loadLabeling()));
	connect(labelingDock, SIGNAL(requestSaveLabeling()),
			chief->labelingModel(), SLOT(saveLabeling()));

	/* Illumination Dock */
	connect(illumDock, SIGNAL(applyIllum()),
			chief->illumModel(), SLOT(applyIllum()));
	connect(illumDock, SIGNAL(illum1Selected(int)),
			chief->illumModel(), SLOT(updateIllum1(int))); //FIXME slot name
	connect(illumDock, SIGNAL(illum2Selected(int)),
			chief->illumModel(), SLOT(updateIllum2(int)));

	// connections between illumDock and dist viewers
	connect(illumDock, SIGNAL(showIlluminationCurveChanged(bool)),
			chief, SIGNAL(showIlluminationCurve(bool)));

	/* Unsupervised Segmentation Dock */
#ifdef WITH_SEG_MEANSHIFT
	ClusteringModel const*cm = chief->clusteringModel();
	clusteringDock->setNumBands(nbands);
	connect(chief->imageModel(), SIGNAL(numBandsROIChanged(int)),
			clusteringDock, SLOT(setNumBands(int)));
	connect(cm, SIGNAL(progressChanged(int)),
			clusteringDock, SLOT(updateProgress(int)));
	connect(cm, SIGNAL(segmentationCompleted()),
			clusteringDock, SLOT(processSegmentationCompleted()));
	connect(clusteringDock, SIGNAL(segmentationRequested(vole::Command*,int,bool)),
			cm, SLOT(startSegmentation(vole::Command*,int,bool)));
	connect(clusteringDock, SIGNAL(cancelSegmentationRequested()),
			cm, SLOT(cancel()));
	connect(cm, SIGNAL(setLabelsRequested(cv::Mat1s)),
			chief->labelingModel(), SLOT(setLabels(cv::Mat1s)));

	// FIXME hide for release?
	//clusteringDock->hide();
#endif /* WITH_SEG_MEANSHIFT */

	/* Normalization Dock */
	connect(chief->imageModel(),
			SIGNAL(observedDataRangeUdpate(representation::t, multi_img::Range)),
			normDock,
			SLOT(setNormRange(representation::t,multi_img::Range)));
	connect(normDock,
			SIGNAL(normalizationParametersChanged(
					   representation::t,multi_img::NormMode,multi_img::Range)),
			chief->imageModel(),
			SLOT(setNormalizationParameters(
					 representation::t,multi_img::NormMode,multi_img::Range)));
	connect(normDock, SIGNAL(applyNormalizationRequested()),
			chief,
			SLOT(invalidateROI()));

	/* Label Dock */
	connect(chief->labelingModel(),
			SIGNAL(newLabeling(cv::Mat1s,QVector<QColor>,bool)),
			labelDock, SLOT(setLabeling(cv::Mat1s,QVector<QColor>,bool)));
	connect(chief->labelingModel(),
			SIGNAL(partialLabelUpdate(const cv::Mat1s&, const cv::Mat1b&)),
			labelDock,
			SLOT(processPartialLabelUpdate(const cv::Mat1s&,const cv::Mat1b&)));
	connect(labelDock, SIGNAL(mergeLabelsRequested(QVector<int>)),
			chief->labelingModel(), SLOT(mergeLabels(QVector<int>)));
	connect(labelDock, SIGNAL(deleteLabelsRequested(QVector<int>)),
			chief->labelingModel(), SLOT(deleteLabels(QVector<int>)));
	connect(labelDock, SIGNAL(consolidateLabelsRequested()),
			chief->labelingModel(), SLOT(consolidate()));
	connect(labelDock, SIGNAL(highlightLabelRequested(short,bool)),
			this, SLOT(highlightSingleLabel(short,bool)));
	connect(labelDock, SIGNAL(highlightLabelRequested(short,bool)),
			bandDock->bandView(), SLOT(highlightSingleLabel(short,bool)));
	connect(labelDock, SIGNAL(labelMaskIconsRequested()),
			chief->labelingModel(), SLOT(computeLabelIcons()));
	connect(labelDock, SIGNAL(labelMaskIconSizeChanged(const QSize&)),
			chief->labelingModel(), SLOT(setLabelIconSize(const QSize&)));
	connect(chief->labelingModel(),
			SIGNAL(labelIconsComputed(const QVector<QImage>&)),
			labelDock, SLOT(processMaskIconsComputed(const QVector<QImage>&)));
}

void DockController::setGUIEnabled(bool enable, TaskType tt)
{
	bandDock->setEnabled(enable);
//TODO	bandDock->bandView()->setEnabled(enable);

	if (tt == TT_SELECT_ROI && (!enable)) {
		/* TODO: check if this is enough to make sure no label changes
		 * happen during ROI recomputation */
		bandDock->bandView()->commitLabelChanges();
	}

	//TODO
	//	labelDock->setEnabled(enable);
	falseColorDock->setEnabled(enable);

	// TODO limitedMode - availabe from Controller?
	//illumDock->setEnabled((enable || tt == TT_APPLY_ILLUM) && !im->isLimitedMode());
	illumDock->setEnabled((enable || tt == TT_APPLY_ILLUM));

#ifdef WITH_SEG_MEANSHIFT
	clusteringDock->setEnabled(enable && !chief->imageModel()->isLimitedMode());
#endif
	roiDock->setEnabled(enable || tt == TT_SELECT_ROI);
	labelDock->setEnabled(enable);
}

void DockController::requestGraphseg(representation::t repr,
									 const vole::GraphSegConfig &config,
									 bool resetLabel)
{
	cv::Mat1s seedMap = bandDock->bandView()->getSeedMap();
	emit requestGraphseg(repr, seedMap, config, resetLabel);
}

void DockController::requestGraphsegCurBand(const vole::GraphSegConfig &config,
											bool resetLabel)
{
	representation::t repr = bandDock->getCurRepresentation();
	int bandId = bandDock->getCurBandId();
	cv::Mat1s seedMap = bandDock->bandView()->getSeedMap();
	emit requestGraphsegBand(repr, bandId, seedMap, config, resetLabel);
}

void DockController::highlightSingleLabel(short label, bool highlight)
{
	/* currently we only support a single label highlight. a negative signal
	 * will therefore deactivate all highlights. */
	if (highlight) {
		emit toggleSingleLabel(true);
		emit singleLabelSelected(label);
	} else {
		emit toggleSingleLabel(false);
	}
}
