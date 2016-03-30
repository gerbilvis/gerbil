
#include "controller.h"
#include "distviewcontroller.h"

#include <widgets/mainwindow.h>
#include <widgets/graphsegwidget.h>

#include <docks/banddock.h>
#include <widgets/bandview.h>
#include <docks/normdock.h>
#include <docks/falsecolordock.h>
#include <docks/roidock.h>
#include <docks/illumdock.h>
#include <docks/clusteringdock.h>
#include <docks/labeldock.h>

#include <model/imagemodel.h>
#include <model/labelingmodel.h>
#include <model/falsecolormodel.h>
#include <model/graphsegmentationmodel.h>
#include <model/clusteringmodel.h>
#include <model/illuminationmodel.h>

#include <QSettings>
#include <gerbil_gui_debug.h>


void Controller::initDocks()
{
	createDocks();
	setupDocks();

	QSettings settings;

	/// left side
	mainWindow()->addDockWidget(Qt::LeftDockWidgetArea, roiDock);
	mainWindow()->addDockWidget(Qt::LeftDockWidgetArea, illumDock);
	mainWindow()->addDockWidget(Qt::LeftDockWidgetArea, normDock);
	mainWindow()->tabifyDockWidget(illumDock, normDock);
#ifdef WITH_SEG_MEANSHIFT
	mainWindow()->addDockWidget(Qt::LeftDockWidgetArea, clusteringDock);
#endif

	/// right side
	mainWindow()->addDockWidget(Qt::RightDockWidgetArea, bandDock);
	mainWindow()->addDockWidget(Qt::RightDockWidgetArea, labelDock);
	mainWindow()->addDockWidget(Qt::RightDockWidgetArea, falseColorDock);

	// dock arrangement
	//	mainWindow()->tabifyDockWidget(roiDock, falseColorDock);
#ifdef WITH_SEG_MEANSHIFT
	//	mainWindow()->tabifyDockWidget(roiDock, clusteringDock);
#endif
	roiDock->raise();
	// restore dock widget positions
	mainWindow()->restoreState(
				settings.value("mainWindow/windowState").toByteArray());

	imageModel()->computeFullRgb();
}

void Controller::createDocks()
{
	assert(NULL != mainWindow());

	bandDock = new BandDock(imageModel()->getFullImageRect(),
							mainWindow());
	normDock = new NormDock(mainWindow());
	roiDock = new RoiDock(mainWindow());
	illumDock = new IllumDock(mainWindow());
	falseColorDock = new FalseColorDock(mainWindow());
#ifdef WITH_SEG_MEANSHIFT
	clusteringDock = new ClusteringDock(mainWindow());
#endif
	labelDock = new LabelDock(mainWindow());
}

void Controller::setupDocks()
{

	/* im -> others */
	connect(imageModel(),
			SIGNAL(bandUpdate(representation::t, int, QPixmap, QString)),
			bandDock,
			SLOT(changeBand(representation::t, int, QPixmap, QString)));

	/* Band Dock */
	connect(labelingModel(), SIGNAL(partialLabelUpdate(const cv::Mat1s&,const cv::Mat1b&)),
			bandDock, SLOT(processLabelingChange(cv::Mat1s,cv::Mat1b)));
	connect(labelingModel(), SIGNAL(newLabeling(cv::Mat1s,QVector<QColor>,bool)),
			bandDock, SLOT(processLabelingChange(cv::Mat1s,QVector<QColor>,bool)));

	connect(dvc, SIGNAL(bandSelected(representation::t, int)),
			bandDock, SLOT(processBandSelected(representation::t,int)));
	connect(bandDock, SIGNAL(subscribeImageBand(QObject *,representation::t,int)),
			this, SLOT(subscribeImageBand(QObject *,representation::t,int)));
	connect(bandDock, SIGNAL(unsubscribeImageBand(QObject *,representation::t,int)),
			this, SLOT(unsubscribeImageBand(QObject *,representation::t,int)));

	connect(bandDock->bandView(), SIGNAL(alteredLabels(cv::Mat1s,cv::Mat1b)),
			labelingModel(), SLOT(alterPixels(cv::Mat1s,cv::Mat1b)));
	connect(bandDock->bandView(), SIGNAL(newLabeling(cv::Mat1s)),
			labelingModel(), SLOT(setLabels(cv::Mat1s)));

	connect(bandDock->bandView(), SIGNAL(pixelOverlay(int,int)),
			this, SIGNAL(requestPixelOverlay(int,int)));
	connect(bandDock->bandView(), SIGNAL(labelSelected(int)),
			this, SIGNAL(labelSelected(int)));
	connect(bandDock, SIGNAL(currentLabelChanged(int)),
			this, SIGNAL(currentLabelChanged(int)));
	// alterLabel(short) -> clear label
	connect(bandDock, SIGNAL(clearLabelRequested(short)),
			labelingModel(), SLOT(alterLabel(short)));
	connect(bandDock, SIGNAL(newLabelRequested()),
			labelingModel(), SLOT(addLabel()));

	connect(bandDock->bandView(), SIGNAL(mergeLabelsRequested(QVector<int>)),
			labelDock, SIGNAL(mergeLabelsRequested(QVector<int>)));

	/* Graph Segmentation Widget */
	// Controller adds missing information and resends the signal
	connect(bandDock->graphSegWidget(),
			SIGNAL(requestGraphseg(representation::t,
								   seg_graphs::GraphSegConfig,bool)),
			this,
			SLOT(requestGraphseg(representation::t,
								 seg_graphs::GraphSegConfig,bool)));
	connect(this,
			SIGNAL(requestGraphseg(representation::t,cv::Mat1s,
								   seg_graphs::GraphSegConfig,bool)),
			graphSegmentationModel(),
			SLOT(runGraphseg(representation::t,cv::Mat1s,
							 seg_graphs::GraphSegConfig,bool)));
	connect(bandDock->graphSegWidget(),
			SIGNAL(requestGraphsegCurBand(const seg_graphs::GraphSegConfig &,bool)),
			this,
			SLOT(requestGraphsegCurBand(const seg_graphs::GraphSegConfig &,bool)));
	connect(this,
			SIGNAL(requestGraphsegBand(representation::t,int,cv::Mat1s,
									   const seg_graphs::GraphSegConfig &,bool)),
			graphSegmentationModel(),
			SLOT(runGraphsegBand(representation::t,int,cv::Mat1s,
								 const seg_graphs::GraphSegConfig &,bool)));

	// GraphSegModel -> BandDock
	connect(bandDock, SIGNAL(currentLabelChanged(int)),
			graphSegmentationModel(), SLOT(setCurLabel(int)));
	connect(graphSegmentationModel(), SIGNAL(seedingDone()),
			bandDock->graphSegWidget(), SLOT(processSeedingDone()));

	connect(this, SIGNAL(requestOverlay(const cv::Mat1b&)),
			bandDock->bandView(), SLOT(drawOverlay(const cv::Mat1b&)));

	connect(this, SIGNAL(toggleIgnoreLabels(bool)),
			bandDock->bandView(), SLOT(toggleShowLabels(bool)));

	/* FalseColor Dock */
	connect(falseColorDock, SIGNAL(subscribeFalseColoring(QObject*, FalseColoring::Type)),
			this, SLOT(subscribeFalseColor(QObject*, FalseColoring::Type)));
	connect(falseColorDock, SIGNAL(unsubscribeFalseColoring(QObject*, FalseColoring::Type)),
			this, SLOT(unsubscribeFalseColor(QObject*,FalseColoring::Type)));
	connect(falseColorDock, SIGNAL(falseColoringRecalcRequested(FalseColoring::Type)),
			this, SLOT(recalcFalseColor(FalseColoring::Type)));
	connect(falseColorDock, SIGNAL(pixelOverlay(int,int)),
	        this, SIGNAL(requestPixelOverlay(int,int)));

	connect(falseColorModel(), SIGNAL(progressChanged(FalseColoring::Type,int)),
			falseColorDock, SLOT(processCalculationProgressChanged(FalseColoring::Type,int)));
	connect(falseColorModel(), SIGNAL(falseColoringUpdate(FalseColoring::Type,QPixmap)),
			falseColorDock, SLOT(processFalseColoringUpdate(FalseColoring::Type,QPixmap)));
	connect(falseColorModel(), SIGNAL(computationCancelled(FalseColoring::Type)),
			falseColorDock, SLOT(processComputationCancelled(FalseColoring::Type)));

	connect(this, SIGNAL(pendingFalseColorUpdate(FalseColoring::Type)),
			falseColorDock, SLOT(setCalculationInProgress(FalseColoring::Type)));

	// needed for ROI dock, clustering dock
	int nbands = imageModel()->getNumBandsFull();

	/* ROI Dock */
	roiDock->setMaxBands(nbands);
	// model to dock (reset handled in RoiDock)
	connect(imageModel(), SIGNAL(fullRgbUpdate(QPixmap)),
			roiDock, SLOT(updatePixmap(QPixmap)));

	connect(imageModel(), SIGNAL(roiRectChanged(cv::Rect)),
			roiDock, SLOT(setRoi(cv::Rect)));

	// dock to controller
	connect(roiDock, SIGNAL(roiRequested(const cv::Rect&)),
			this, SLOT(spawnROI(const cv::Rect&)));
	connect(roiDock, SIGNAL(specRescaleRequested(int)),
			this, SLOT(rescaleSpectrum(int)));

	/* Illumination Dock */
	connect(illumDock, SIGNAL(applyIllum()),
			illumModel(), SLOT(applyIllum()));
	connect(illumDock, SIGNAL(illum1Selected(int)),
			illumModel(), SLOT(updateIllum1(int))); //FIXME slot name
	connect(illumDock, SIGNAL(illum2Selected(int)),
			illumModel(), SLOT(updateIllum2(int)));

	// connections between illumDock and dist viewers
	connect(illumDock, SIGNAL(showIlluminationCurveChanged(bool)),
			this, SIGNAL(showIlluminationCurve(bool)));

	/* Unsupervised Segmentation Dock */
#ifdef WITH_SEG_MEANSHIFT
	ClusteringModel const*cm = clusteringModel();
	connect(cm, SIGNAL(progressChanged(int)),
			clusteringDock, SLOT(updateProgress(int)));
	connect(cm, SIGNAL(segmentationCompleted()),
			clusteringDock, SLOT(processSegmentationCompleted()));
	connect(clusteringDock,
			SIGNAL(segmentationRequested(const ClusteringRequest&)),
			cm, SLOT(requestSegmentation(const ClusteringRequest&)));
	connect(clusteringDock, SIGNAL(cancelSegmentationRequested()),
			cm, SLOT(cancel()));
	connect(cm, SIGNAL(setLabelsRequested(cv::Mat1s)),
			labelingModel(), SLOT(setLabels(cv::Mat1s)));

#endif /* WITH_SEG_MEANSHIFT */

	/* Normalization Dock */
	connect(imageModel(),
			SIGNAL(observedDataRangeUdpate(representation::t, multi_img::Range)),
			normDock,
			SLOT(setNormRange(representation::t,multi_img::Range)));
	connect(normDock,
			SIGNAL(normalizationParametersChanged(
					   representation::t,multi_img::NormMode,multi_img::Range)),
			imageModel(),
			SLOT(setNormalizationParameters(
					 representation::t,multi_img::NormMode,multi_img::Range)));
	connect(normDock, SIGNAL(applyNormalizationRequested()),
			this, SLOT(invalidateROI()));

	/* Label Dock */
	labelDock->setImageSize(imgSize);
	connect(labelingModel(),
			SIGNAL(newLabeling(cv::Mat1s,QVector<QColor>,bool)),
			labelDock, SLOT(setLabeling(cv::Mat1s,QVector<QColor>,bool)));
	connect(labelingModel(),
			SIGNAL(partialLabelUpdate(const cv::Mat1s&, const cv::Mat1b&)),
			labelDock,
			SLOT(processPartialLabelUpdate(const cv::Mat1s&,const cv::Mat1b&)));
	connect(labelDock, SIGNAL(mergeLabelsRequested(QVector<int>)),
			labelingModel(), SLOT(mergeLabels(QVector<int>)));
	connect(labelDock, SIGNAL(deleteLabelsRequested(QVector<int>)),
			labelingModel(), SLOT(deleteLabels(QVector<int>)));
	connect(labelDock, SIGNAL(consolidateLabelsRequested()),
			labelingModel(), SLOT(consolidate()));
	connect(labelDock, SIGNAL(toggleLabelHighlightRequested(short)),
			this, SLOT(toggleLabelHighlight(short)));
	connect(labelDock, SIGNAL(toggleLabelHighlightRequested(short)),
			bandDock->bandView(), SLOT(toggleLabelHighlight(short)));
	connect(labelDock, SIGNAL(labelMaskIconsRequested(QSize)),
			labelingModel(), SLOT(computeLabelIcons(QSize)));
	connect(labelingModel(),
			SIGNAL(labelIconsComputed(QVector<QImage>)),
			labelDock, SLOT(processMaskIconsComputed(QVector<QImage>)));
	connect(labelDock, SIGNAL(applyROIChanged(bool)),
			labelingModel(), SLOT(setApplyROI(bool)));
	connect(labelDock, SIGNAL(requestLoadLabeling()),
			labelingModel(), SLOT(loadLabeling()));
	connect(labelDock, SIGNAL(requestSaveLabeling()),
			labelingModel(), SLOT(saveLabeling()));
	connect(imageModel(), SIGNAL(roiRectChanged(cv::Rect)),
			labelDock, SLOT(processRoiRectChanged(cv::Rect)));

	connect(bandDock->bandView(), SIGNAL(labelSelected(int)),
			labelDock, SLOT(toggleLabelSelection(int)));
}

void Controller::requestGraphseg(representation::t repr,
								 const seg_graphs::GraphSegConfig &config,
								 bool resetLabel)
{
	cv::Mat1s seedMap = bandDock->bandView()->getSeedMap();
	emit requestGraphseg(repr, seedMap, config, resetLabel);
}

void Controller::requestGraphsegCurBand(const seg_graphs::GraphSegConfig &config,
										bool resetLabel)
{
	representation::t repr = bandDock->getCurRepresentation();
	int bandId = bandDock->getCurBandId();
	cv::Mat1s seedMap = bandDock->bandView()->getSeedMap();
	emit requestGraphsegBand(repr, bandId, seedMap, config, resetLabel);
}

void Controller::toggleLabelHighlight(short label)
{
	emit labelSelected(label);
}
