/*
	Copyright(c) 2012 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "mainwindow.h"
#include "controller.h"
#include "iogui.h"
#include "commandrunner.h"
#include "multi_img_tasks.h"
#include "tasks/rgbtbb.h"
#include "tasks/rgbserial.h"
#include "tasks/normrangecuda.h"
#include "tasks/normrangetbb.h"
#include "tasks/graphsegbackground.h"

#include "docks/illumdock.h"
#include "docks/rgbdock.h"
#include "docks/graphsegmentationdock.h"
#include "docks/ussegmentationdock.h"


#include <background_task_queue.h>

#include <labeling.h>
#include <qtopencv.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <QPainter>
#include <QIcon>
#include <QSignalMapper>
#include <iostream>
#include <QShortcut>

MainWindow::MainWindow(bool limitedMode)
	: limitedMode(limitedMode),
	  contextMenu(NULL)
	  //FIXME SEGMENTATION_TODO
//	  ,graphsegResult(new cv::Mat1s())
{
	// create all objects
	setupUi(this);
}

// todo: move to the new bandDock
void MainWindow::clearLabelOrSeeds()
{
	if (bandView->isSeedModeEnabled()) {
		bandView->clearSeeds();
	} else {
		emit clearLabelRequested(bandView->getCurLabel());
	}
}

void MainWindow::addToLabel()
{
	cv::Mat1b mask = viewerContainer->getHighlightMask();
	emit alterLabelRequested(bandView->getCurLabel(), mask, false);
}

void MainWindow::remFromLabel()
{
	cv::Mat1b mask = viewerContainer->getHighlightMask();
	emit alterLabelRequested(bandView->getCurLabel(), mask, true);
}

// todo: move to the new bandDock
void MainWindow::changeBand(QPixmap band, QString desc)
{
	bandView->setEnabled(true);
	bandView->setPixmap(band);
	bandDock->setWindowTitle(desc);
}

// todo: move to the new bandDock
void MainWindow::processLabelingChange(const cv::Mat1s &labels,
									   const QVector<QColor> &colors,
									   bool colorsChanged)
{
	if (!colors.empty()) {
		// use colors for our awesome label menu (rebuild everything)
		markerSelector->clear();
		for (int i = 1; i < colors.size(); ++i) // 0 is index for unlabeled
		{
			markerSelector->addItem(colorIcon(colors.at(i)), "");
		}
		markerSelector->addItem(QIcon(":/toolbar/add"), "");
	}

	// tell bandview about the update as well
	bandView->updateLabeling(labels, colors, colorsChanged);
}

// todo: move to the new bandDock
void MainWindow::processLabelingChange(const cv::Mat1s &labels,
									   const cv::Mat1b &mask)
{
	// tell bandview about the update
	bandView->updateLabeling(labels, mask);
}

// todo: move to the new bandDock
void MainWindow::selectLabel(int index)
{
	// markerSelector has no label zero, therefore off by one
	markerSelector->setCurrentIndex(index - 1);
}

void MainWindow::initUI(cv::Rect dim, size_t size)
{
	// used in loadSeeds(), maybe also for showing generic metadata
	dimensions = dim;

	/* init bandsSlider */
	bandsLabel->setText(QString("%1 bands").arg(size));
	bandsSlider->setMinimum(3);
	bandsSlider->setMaximum(size);
	bandsSlider->setValue(size);

	// FIXME SEGMENTATION_TODO
	// TODO: these will be docks
	//initGraphsegUI()

	initNormalizationUI();

	// FIXME SEGMENTATION_TODO
//	/* more manual work to get GUI in proper shape */
//	graphsegWidget->hide();

	viewerContainer->initUi();
}

void MainWindow::initSignals(Controller *chief)
{
	/* slots & signals: GUI only */
	connect(docksButton, SIGNAL(clicked()),
			this, SLOT(openContextMenu()));

	/* TODO: this all belongs to new banddock */

//	we decided to remove this functionality for now
//	connect(bandDock, SIGNAL(topLevelChanged(bool)),
//			this, SLOT(reshapeDock(bool)));

	connect(markerSelector, SIGNAL(currentIndexChanged(int)),
			bandView, SLOT(changeCurrentLabel(int)));

	connect(ignoreButton, SIGNAL(toggled(bool)),
			markButton, SLOT(setDisabled(bool)));
	connect(ignoreButton, SIGNAL(toggled(bool)),
			nonmarkButton, SLOT(setDisabled(bool)));
	connect(ignoreButton, SIGNAL(toggled(bool)),
			singleButton, SLOT(setDisabled(bool)));
	connect(ignoreButton, SIGNAL(toggled(bool)),
			bandView, SLOT(toggleShowLabels(bool)));
	connect(singleButton, SIGNAL(toggled(bool)),
			bandView, SLOT(toggleSingleLabel(bool)));
	connect(alphaSlider, SIGNAL(valueChanged(int)),
			bandView, SLOT(applyLabelAlpha(int)));


	/* labeling manipulation triggers */
	connect(clearButton, SIGNAL(clicked()),
			this, SLOT(clearLabelOrSeeds()));

	connect(bandView, SIGNAL(newLabel()),
			chief, SLOT(addLabel()));


	// labeldock
	connect(lLoadButton, SIGNAL(clicked()),
			chief, SLOT(loadLabeling()));
	connect(lSaveButton, SIGNAL(clicked()),
			chief, SLOT(saveLabeling()));
	connect(lLoadSeedButton, SIGNAL(clicked()),
			this, SLOT(loadSeeds()));


	// for viewports
	connect(ignoreButton, SIGNAL(toggled(bool)),
			chief, SLOT(toggleLabels(bool)));

	// label manipulation fuckup
	connect(addButton, SIGNAL(clicked()),
			this, SLOT(addToLabel()));
	connect(remButton, SIGNAL(clicked()),
			this, SLOT(remFromLabel()));

	connect(viewerContainer, SIGNAL(drawOverlay(const cv::Mat1b&)),
			bandView, SLOT(drawOverlay(const cv::Mat1b&)));

	// todo: we connect it here as we disconnect it here as well. we will change
	// that.
	connect(&bandView->labelTimer, SIGNAL(timeout()),
			bandView, SLOT(commitLabelChanges()));

	connect(bandView, SIGNAL(alteredLabels(const cv::Mat1s&, const cv::Mat1b&)),
			this, SIGNAL(alterLabelingRequested(cv::Mat1s,cv::Mat1b)));
	connect(bandView, SIGNAL(newLabeling(const cv::Mat1s&)),
			this, SIGNAL(newLabelingRequested(cv::Mat1s)));

	/* when applybutton is pressed, bandView commits full label matrix */
	connect(applyButton, SIGNAL(clicked()),
			bandView, SLOT(commitLabels()));

	connect(bandView, SIGNAL(killHover()),
			viewerContainer, SIGNAL(viewportsKillHover()));
	connect(bandView, SIGNAL(pixelOverlay(int, int)),
			viewerContainer, SIGNAL(viewersOverlay(int, int)));
	connect(bandView, SIGNAL(newSingleLabel(short)),
			viewerContainer, SIGNAL(viewersHighlight(short)));

	connect(markButton, SIGNAL(toggled(bool)),
			viewerContainer, SIGNAL(viewersToggleLabeled(bool)));
	connect(nonmarkButton, SIGNAL(toggled(bool)),
			viewerContainer, SIGNAL(viewersToggleUnlabeled(bool)));

	connect(viewerContainer, SIGNAL(normTargetChanged(bool)),
			this, SLOT(normTargetChanged(bool)));

	connect(viewerContainer, SIGNAL(requestGUIEnabled(bool,TaskType)),
			this, SLOT(setGUIEnabled(bool,TaskType)));
	connect(viewerContainer, SIGNAL(requestGUIEnabled(bool,TaskType)),
			this, SLOT(debugRequestGUIEnabled(bool,TaskType)));
	connect(viewerContainer, SIGNAL(viewportAddSelection()),
			this, SLOT(addToLabel()));
	connect(viewerContainer, SIGNAL(viewportRemSelection()),
			this, SLOT(remFromLabel()));

	connect(bandsSlider, SIGNAL(valueChanged(int)),
			this, SLOT(bandsSliderMoved(int)));
	connect(bandsSlider, SIGNAL(sliderMoved(int)),
			this, SLOT(bandsSliderMoved(int)));
	connect(this, SIGNAL(specRescaleRequested(size_t)),
			chief, SLOT(rescaleSpectrum(size_t)));

	/// global shortcuts
	QShortcut *scr = new QShortcut(Qt::CTRL + Qt::Key_S, this);
	connect(scr, SIGNAL(activated()), this, SLOT(screenshot()));

	/* now that we are connected, humbly request RGB image for roiView */
	// FIXME no, this can be requested by the rgb dock itself.
	//emit rgbRequested();

// old
//	connect(graphsegButton, SIGNAL(toggled(bool)),
//			graphsegWidget, SLOT(setVisible(bool)));
//	connect(graphsegButton, SIGNAL(toggled(bool)),
//			bandView, SLOT(toggleSeedMode(bool)));
// new
	connect(graphsegButton, SIGNAL(toggled(bool)),
			this, SIGNAL(graphSegDockVisibleRequested(bool)));
	connect(graphsegButton, SIGNAL(toggled(bool)),
			bandView, SLOT(toggleSeedMode(bool)));
}

void MainWindow::setGUIEnabled(bool enable, TaskType tt)
{
	/** for enable, this just re-enables everything
	 * for disable, this typically disables everything except the sender, so
	 * that the user can re-decide on that aspect or sth.
	 * it is a bit strange
	 */
	bandsSlider->setEnabled(enable || tt == TT_BAND_COUNT);
	ignoreButton->setEnabled(enable || tt == TT_TOGGLE_LABELS);
	addButton->setEnabled(enable);
	remButton->setEnabled(enable);

	viewerContainer->setGUIEnabled(enable, tt);

	applyButton->setEnabled(enable);
	clearButton->setEnabled(enable);
	bandView->setEnabled(enable);

	normDock->setEnabled((enable || tt == TT_NORM_RANGE || tt == TT_CLAMP_RANGE_IMG || tt == TT_CLAMP_RANGE_GRAD) && !limitedMode);
	normIButton->setEnabled(enable || tt == TT_NORM_RANGE || tt == TT_CLAMP_RANGE_IMG);
	normGButton->setEnabled(enable || tt == TT_NORM_RANGE || tt == TT_CLAMP_RANGE_GRAD);
	normModeBox->setEnabled(enable);
	normApplyButton->setEnabled(enable || tt == TT_NORM_RANGE);
	normClampButton->setEnabled(enable || tt == TT_CLAMP_RANGE_IMG || tt == TT_CLAMP_RANGE_GRAD);

	emit requestEnableDocks(enable, tt);

	if (tt == TT_SELECT_ROI && (!enable)) {
		/* TODO: check if this is enough to make sure no label changes
		 * happen during ROI recomputation */
		bandView->commitLabelChanges();
	}
}

// TODO: controller
void MainWindow::bandsSliderMoved(int b)
{
	bandsLabel->setText(QString("%1 bands").arg(b));
	if (!bandsSlider->isSliderDown()) {
		emit specRescaleRequested(b);
	}
}

void MainWindow::debugRequestGUIEnabled(bool enable, TaskType tt)
{
	//GGDBG_CALL();
	//GGDBGM(format("enable=%1%, tt=%2%\n") %enable %tt)
}

void MainWindow::initNormalizationUI()
{
	normModeBox->addItem("Observed");
	normModeBox->addItem("Theoretical");
	normModeBox->addItem("Fixed");
	connect(normIButton, SIGNAL(toggled(bool)),
			this, SLOT(normTargetChanged()));
	connect(normGButton, SIGNAL(toggled(bool)),
			this, SLOT(normTargetChanged()));
	connect(normModeBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(normModeSelected(int)));
	connect(normMinBox, SIGNAL(valueChanged(double)),
			this, SLOT(normModeFixed()));
	connect(normMaxBox, SIGNAL(valueChanged(double)),
			this, SLOT(normModeFixed()));
	connect(normApplyButton, SIGNAL(clicked()),
			this, SLOT(applyNormUserRange()));
	connect(normClampButton, SIGNAL(clicked()),
			this, SLOT(clampNormUserRange()));
}

void MainWindow::normTargetChanged(bool usecurrent)
{
/*
	// reset gui to current settings
	int target = (normIButton->isChecked() ? 0 : 1);
	MultiImg::NormMode m = (target == 0 ? normIMG : normGRAD);

	// update normModeBox
	normModeBox->setCurrentIndex(m);

	// update norm range spin boxes
	normModeSelected(m, true, usecurrent);
	*/
}

void MainWindow::normModeSelected(int mode, bool targetchange, bool usecurrent)
{
	/*
	MultiImg::NormMode nm = static_cast<MultiImg::NormMode>(mode);
	if (nm == MultiImg::NORM_FIXED && !targetchange) // user edits from currenty viewed values
		return;

	int target = (normIButton->isChecked() ? 0 : 1);

	if (!usecurrent) {
		multi_img::Value min;
		multi_img::Value max;
		if (target == 0) {
			SharedDataLock hlock(normIMGRange->mutex);
			min = (*normIMGRange)->first;
			max = (*normIMGRange)->second;
		} else {
			SharedDataLock hlock(normGRADRange->mutex);
			min = (*normGRADRange)->first;
			max = (*normGRADRange)->second;
		}

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_DATARANGE) {
			BackgroundTaskPtr taskNormRange(new NormRangeCuda(
				(target == 0 ? image : gradient), 
				(target == 0 ? normIMGRange : normGRADRange), 
				nm, target, min, max, false));
			queue.push(taskNormRange);
			taskNormRange->wait();
		} else {
			BackgroundTaskPtr taskNormRange(new NormRangeTbb(
				(target == 0 ? image : gradient), 
				(target == 0 ? normIMGRange : normGRADRange), 
				nm, target, min, max, false));
			queue.push(taskNormRange);
			taskNormRange->wait();
		}
	}

	double min;
	double max;
	if (target == 0) {
		SharedDataLock hlock(normIMGRange->mutex);
		min = (*normIMGRange)->first;
		max = (*normIMGRange)->second;
	} else {
		SharedDataLock hlock(normGRADRange->mutex);
		min = (*normGRADRange)->first;
		max = (*normGRADRange)->second;
	}

	// prevent signal loop
	normMinBox->blockSignals(true);
	normMaxBox->blockSignals(true);
	normMinBox->setValue(min);
	normMaxBox->setValue(max);
	normMinBox->blockSignals(false);
	normMaxBox->blockSignals(false);
	*/
}

void MainWindow::normModeFixed()
{
	if (normModeBox->currentIndex() != 2)
		normModeBox->setCurrentIndex(2);
}

void MainWindow::applyNormUserRange()
{
	/*
	representation::t target = (normIButton->isChecked() ?
							representation::IMG : representation::GRAD);

	// set internal norm mode
	MultiImg::NormMode &nm = (target == 0 ? normIMG : normGRAD);
	nm = static_cast<MultiImg::NormMode>(normModeBox->currentIndex());

	queue.cancelTasks();
	setGUIEnabled(false, TT_NORM_RANGE);

	// if available, overwrite with more precise values than in the spin boxes.
	if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_DATARANGE) {
		BackgroundTaskPtr taskNormRange(new NormRangeCuda(
			(target == 0 ? image : gradient), 
			(target == 0 ? normIMGRange : normGRADRange), 
			(target == 0 ? normIMG : normGRAD), target, normMinBox->value(), normMaxBox->value(), true));
		queue.push(taskNormRange);
	} else {
		BackgroundTaskPtr taskNormRange(new NormRangeTbb(
			(target == 0 ? image : gradient), 
			(target == 0 ? normIMGRange : normGRADRange), 
			(target == 0 ? normIMG : normGRAD), target, normMinBox->value(), normMaxBox->value(), true));
		queue.push(taskNormRange);
	}

	// re-initialize gui (duplication from applyROI())
	if (target == 0) {
//		viewIMG->updateBinning(-1);
		viewerContainer->updateBinning(representation::IMG,-1);

		BackgroundTaskPtr taskFinishNorm(new BackgroundTask());
		QObject::connect(taskFinishNorm.get(), SIGNAL(finished(bool)), 
			viewerContainer, SLOT(finishNormRangeImgChange(bool)), Qt::QueuedConnection);
		queue.push(taskFinishNorm);
	} else {
//		viewGRAD->updateBinning(-1);
		viewerContainer->updateBinning(GRAD,-1);

		BackgroundTaskPtr taskFinishNorm(new BackgroundTask());
		QObject::connect(taskFinishNorm.get(), SIGNAL(finished(bool)), 
			viewerContainer, SLOT(finishNormRangeGradChange(bool)), Qt::QueuedConnection);
		queue.push(taskFinishNorm);
	}

	BackgroundTaskPtr taskEpilog(new BackgroundTask());
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
		this, SLOT(finishTask(bool)), Qt::QueuedConnection);
	queue.push(taskEpilog);
	*/
}

void MainWindow::clampNormUserRange()
{
	/*
	int target = (normIButton->isChecked() ? 0 : 1);

	// set internal norm mode
	MultiImg::NormMode &nm = (target == 0 ? normIMG : normGRAD);
	nm = static_cast<MultiImg::NormMode>(normModeBox->currentIndex());

	/// if image is changed, change full image. for gradient, we cannot preserve
	///	the gradient over ROI or illuminant changes, so it remains a local change
	if (target == 0) {
		queue.cancelTasks(roi);
		setGUIEnabled(false, TT_CLAMP_RANGE_IMG);

		// if available, overwrite with more precise values than in the spin boxes.
		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_DATARANGE) {
			BackgroundTaskPtr taskNormRange(new NormRangeCuda(
				(target == 0 ? image : gradient), 
				(target == 0 ? normIMGRange : normGRADRange), 
				(target == 0 ? normIMG : normGRAD), target, normMinBox->value(), normMaxBox->value(), true, roi));
			queue.push(taskNormRange);
		} else {
			BackgroundTaskPtr taskNormRange(new NormRangeTbb(
				(target == 0 ? image : gradient), 
				(target == 0 ? normIMGRange : normGRADRange), 
				(target == 0 ? normIMG : normGRAD), target, normMinBox->value(), normMaxBox->value(), true, roi));
			queue.push(taskNormRange);
		}

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_CLAMP) {
			BackgroundTaskPtr taskClamp(new MultiImg::ClampCuda(
				image_lim, image, roi, false));
			queue.push(taskClamp);
		} else {
			BackgroundTaskPtr taskClamp(new MultiImg::ClampTbb(
				image_lim, image, roi, false));
			queue.push(taskClamp);
		}

		applyROI(false);
		rgbDock->setEnabled(false);

		// create task to compute rgb image in parallel
		BackgroundTaskPtr taskRgb(new RgbTbb(
			image_lim, mat3f_ptr(new SharedData<cv::Mat3f>(new cv::Mat3f)), full_rgb_temp, roi));
		// connect finished signal to our update slot
		// the task will then trigger the GUI update to show computation result
		QObject::connect(taskRgb.get(), SIGNAL(finished(bool)), this, SLOT(updateRGB(bool)), Qt::QueuedConnection);
		// after signal is connected, enqueue the task in our taskqueue
		queue.push(taskRgb);

		BackgroundTaskPtr taskEpilog(new BackgroundTask(roi));
		QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
			this, SLOT(finishTask(bool)), Qt::QueuedConnection);
		queue.push(taskEpilog);
	} else {
		queue.cancelTasks();
		setGUIEnabled(false, TT_CLAMP_RANGE_GRAD);

		// if available, overwrite with more precise values than in the spin boxes.
		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_DATARANGE) {
			BackgroundTaskPtr taskNormRange(new NormRangeCuda(
				(target == 0 ? image : gradient), 
				(target == 0 ? normIMGRange : normGRADRange), 
				(target == 0 ? normIMG : normGRAD), target, normMinBox->value(), normMaxBox->value(), true));
			queue.push(taskNormRange);
		} else {
			BackgroundTaskPtr taskNormRange(new NormRangeTbb(
				(target == 0 ? image : gradient), 
				(target == 0 ? normIMGRange : normGRADRange), 
				(target == 0 ? normIMG : normGRAD), target, normMinBox->value(), normMaxBox->value(), true));
			queue.push(taskNormRange);
		}

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_CLAMP) {
			BackgroundTaskPtr taskClamp(new MultiImg::ClampCuda(gradient, gradient));
			queue.push(taskClamp);
		} else {
			BackgroundTaskPtr taskClamp(new MultiImg::ClampTbb(gradient, gradient));
			queue.push(taskClamp);
		}

//		viewGRAD->updateBinning(-1);
		viewerContainer->updateBinning(GRAD,-1);

		BackgroundTaskPtr taskFinishClamp(new BackgroundTask());
		QObject::connect(taskFinishClamp.get(), SIGNAL(finished(bool)), 
			this, SLOT(finishNormRangeGradChange(bool)), Qt::QueuedConnection);
		queue.push(taskFinishClamp);

		BackgroundTaskPtr taskEpilog(new BackgroundTask());
		QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
			this, SLOT(finishTask(bool)), Qt::QueuedConnection);
		queue.push(taskEpilog);
	}
	*/
}

// todo: part of banddock
//void MainWindow::initGraphsegUI()
//{
//	graphsegSourceBox->addItem("Image", 0);
//	graphsegSourceBox->addItem("Gradient", 1); // TODO PCA
//	graphsegSourceBox->addItem("Shown Band", 2);
//	graphsegSourceBox->setCurrentIndex(0);

//	graphsegSimilarityBox->addItem("Manhattan distance (L1)", vole::MANHATTAN);
//	graphsegSimilarityBox->addItem("Euclidean distance (L2)", vole::EUCLIDEAN);
//	graphsegSimilarityBox->addItem(QString::fromUtf8("Chebyshev distance (L∞)"),
//								   vole::CHEBYSHEV);
//	graphsegSimilarityBox->addItem("Spectral Angle", vole::MOD_SPEC_ANGLE);
//	graphsegSimilarityBox->addItem("Spectral Information Divergence",
//								   vole::SPEC_INF_DIV);
//	graphsegSimilarityBox->addItem("SID+SAM I", vole::SIDSAM1);
//	graphsegSimilarityBox->addItem("SID+SAM II", vole::SIDSAM2);
//	graphsegSimilarityBox->addItem("Normalized L2", vole::NORM_L2);
//	graphsegSimilarityBox->setCurrentIndex(3);

//	graphsegAlgoBox->addItem("Kruskal", vole::KRUSKAL);
//	graphsegAlgoBox->addItem("Prim", vole::PRIM);
//	graphsegAlgoBox->addItem("Power Watershed q=2", vole::WATERSHED2);
//	graphsegAlgoBox->setCurrentIndex(1);

//	connect(graphsegButton, SIGNAL(toggled(bool)),
//			graphsegWidget, SLOT(setVisible(bool)));
//	connect(graphsegButton, SIGNAL(toggled(bool)),
//			bandView, SLOT(toggleSeedMode(bool)));
//	connect(graphsegGoButton, SIGNAL(clicked()),
//			this, SLOT(startGraphseg()));
//	connect(this, SIGNAL(seedingDone(bool)),
//			graphsegButton, SLOT(setChecked(bool)));
//}

//void MainWindow::runGraphseg(SharedMultiImgPtr input,
//							   const vole::GraphSegConfig &config)
//{
//	/*
//	// TODO: why disable GUI? Where is it enabled?
//	setGUIEnabled(false);
//	// TODO: should this be a commandrunner instead? arguable..
//	BackgroundTaskPtr taskGraphseg(new GraphsegBackground(
//		config, input, bandView->seedMap, graphsegResult));
//	QObject::connect(taskGraphseg.get(), SIGNAL(finished(bool)),
//		this, SLOT(finishGraphSeg(bool)), Qt::QueuedConnection);
//	queue.push(taskGraphseg);
//	*/
//}

//void MainWindow::finishGraphSeg(bool success)
//{
//	/*
//	if (success) {
//		// add segmentation to current labeling
//		emit alterLabelRequested(bandView->getCurLabel(),
//								 *(graphsegResult.get()), false);
//		// leave seeding mode for convenience
//		emit seedingDone();
//	}
//	*/
//}

//// TODO: move part of this to controller who obtains image data from imagemodel
//void MainWindow::startGraphseg()
//{
//	/*
//	vole::GraphSegConfig conf("graphseg");
//	conf.algo = (vole::graphsegalg)
//				graphsegAlgoBox->itemData(graphsegAlgoBox->currentIndex())
//				.value<int>();
//	conf.similarity.measure = (vole::similarity_fun)
//	      graphsegSimilarityBox->itemData(graphsegSimilarityBox->currentIndex())
//	      .value<int>();
//#ifdef WITH_EDGE_DETECT
//	conf.som_similarity = false;
//#endif
//	conf.geodesic = graphsegGeodCheck->isChecked();
//	conf.multi_seed = false;
//	int src = graphsegSourceBox->itemData(graphsegSourceBox->currentIndex())
//								 .value<int>();
//	if (src == 0) {
//		runGraphseg(image, conf);
//	} else if (src == 1) {
//		runGraphseg(gradient, conf);
//	} else {	// currently shown band, construct from selection in viewport
//		representation::t type = viewerContainer->getActiveRepresentation();
//		int band = viewerContainer->getSelection(type);
//		SharedMultiImgPtr img = viewerContainer->getViewerImage(type);
//		SharedDataLock img_lock(img->mutex);
//		SharedMultiImgPtr i(new SharedMultiImgBase(
//			new multi_img((**img)[band], (*img)->minval, (*img)->maxval)));
//		img_lock.unlock();
//		runGraphseg(i, conf);
//	}
//	*/
//}

void MainWindow::loadSeeds()
{
	IOGui io("Seed Image File", "seed image", this);
	cv::Mat1s seeding = io.readFile(QString(), 0,
									dimensions.height, dimensions.width);
	if (seeding.empty())
		return;

	bandView->seedMap = seeding;

	// now make sure we are in seed mode
	if (graphsegButton->isChecked()) {
		bandView->refresh();
	} else {
		graphsegButton->toggle();
	}
}

void MainWindow::screenshot()
{
	// grabWindow reads from the display server, so GL parts are not missing
	QPixmap shot = QPixmap::grabWindow(this->winId());

	// we use OpenCV so the user can expect the same data type support
	cv::Mat output = vole::QImage2Mat(shot.toImage());

	IOGui io("Screenshot File", "screenshot", this);
	io.writeFile(QString(), output);
}

QIcon MainWindow::colorIcon(const QColor &color)
{
	QPixmap pm(32, 32);
	pm.fill(color);
	return QIcon(pm);
}

void MainWindow::tabifyDockWidgets(ROIDock *roiDock, RgbDock *rgbDock, IllumDock *illumDock,
		GraphSegmentationDock *graphSegDock, 
		UsSegmentationDock *usSegDock)
{
	// FIXME altmann: IMHO dock arrangement is borked right now.
	// need to decide what goes where.

	// dock arrangement
	tabifyDockWidget(roiDock, rgbDock);
#ifdef WITH_SEG_MEANSHIFT
	tabifyDockWidget(roiDock, usSegDock);
#endif
	roiDock->raise();
	tabifyDockWidget(labelDock, illumDock);
	tabifyDockWidget(labelDock, normDock);
}

void MainWindow::openContextMenu()
{
	delete contextMenu;
	contextMenu = createPopupMenu();
	contextMenu->exec(QCursor::pos());
}

void MainWindow::changeEvent(QEvent *e)
{
	QMainWindow::changeEvent(e);
	switch (e->type()) {
	case QEvent::LanguageChange:
		retranslateUi(this);
		break;
	default:
		break;
	}
}
