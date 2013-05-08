/*
	Copyright(c) 2012 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "mainwindow.h"
#include "iogui.h"
#include "commandrunner.h"
#include "multi_img_tasks.h"
#include "tasks/rgbtbb.h"
#include "tasks/rgbserial.h"
#include "tasks/normrangecuda.h"
#include "tasks/normrangetbb.h"
#include "tasks/graphsegbackground.h"


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

#define USE_CUDA_GRADIENT       1
#define USE_CUDA_DATARANGE      0
#define USE_CUDA_CLAMP          0
#define USE_CUDA_ILLUMINANT     0

MainWindow::MainWindow(BackgroundTaskQueue &queue, multi_img_base *image,
						   QString labelfile, bool limitedMode, QWidget *parent)
	: QMainWindow(parent), queue(queue),
	  startupLabelFile(labelfile), limitedMode(limitedMode),
	  image_lim(new SharedMultiImgBase(image)),
	  image(new SharedMultiImgBase(new multi_img(0, 0, 0))),
	  gradient(new SharedMultiImgBase(new multi_img(0, 0, 0))),
	  //imagepca(new SharedData<multi_img>(new multi_img(0, 0, 0))),
	  //gradientpca(new SharedData<multi_img>(new multi_img(0, 0, 0))),
	  full_labels(image->height, image->width, (short)0),
	  normIMG(MultiImg::NORM_OBSERVED), normGRAD(MultiImg::NORM_OBSERVED),
	  normIMGRange(new SharedData<std::pair<multi_img::Value, multi_img::Value> >(
	  		new std::pair<multi_img::Value, multi_img::Value>(0, 0))),
	  normGRADRange(new SharedData<std::pair<multi_img::Value, multi_img::Value> >(
	  		new std::pair<multi_img::Value, multi_img::Value>(0, 0))),
	  full_rgb_temp(new SharedData<QImage>(new QImage())),
	  usRunner(NULL), contextMenu(NULL),
	  graphsegResult(new multi_img::Mask())
{
	// create all objects
	setupUi(this);
//	viewers[IMG] = viewIMG;
//	viewIMG->setType(IMG);
//	viewers[GRAD] = viewGRAD;
//	viewGRAD->setType(GRAD);
//	viewers[IMGPCA] = viewIMGPCA;
//	viewIMGPCA->setType(IMGPCA);
//	viewers[GRADPCA] = viewGRADPCA;
//	viewGRADPCA->setType(GRADPCA);
//	for (size_t i = 0; i < viewers.size(); ++i)
//		viewers[i]->queue = &queue;

	// do all the bling-bling
	initUI();
	setLabelColors(vole::Labeling::colors(2, true));

	// default to full image if small enough
	roiView->roi = QRect(0, 0, 
		(*image_lim)->width < 513 ? (*image_lim)->width : 512,
		(*image_lim)->height < 513 ? (*image_lim)->height : 512);

	roi = cv::Rect(roiView->roi.x(), roiView->roi.y(), 
		roiView->roi.width(), roiView->roi.height());

	setGUIEnabled(false, TT_SELECT_ROI);

	applyROI(false);

	BackgroundTaskPtr taskEpilog(new BackgroundTask(roi));
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
		this, SLOT(finishTask(bool)), Qt::QueuedConnection);
	queue.push(taskEpilog);

	// load labels
	if (!labelfile.isEmpty()) {
		BackgroundTaskPtr taskLabels(new BackgroundTask(roi));
		QObject::connect(taskLabels.get(), SIGNAL(finished(bool)), 
			this, SLOT(loadLabeling()), Qt::QueuedConnection);
		queue.push(taskLabels);
	}
}

void MainWindow::finishTask(bool success)
{
	if (success) {
		setGUIEnabled(true);
	}
}

void MainWindow::finishROIChange(bool success)
{
	if (success) {
		connect(&bandView->labelTimer, SIGNAL(timeout()), 
			bandView, SLOT(commitLabelChanges()));
	}
}

void MainWindow::applyROI(bool reuse)
{
	for (size_t i = 0; i < viewers.size(); ++i)
		disconnectViewer(i);
	disconnect(&bandView->labelTimer, SIGNAL(timeout()), 
		bandView, SLOT(commitLabelChanges()));

	if (!reuse) {
		SharedDataLock image_lock(image->mutex);
		SharedDataLock gradient_lock(gradient->mutex);
		cv::Rect empty(0, 0, 0, 0);
		(*image)->roi = empty;
		(*gradient)->roi = empty;

		if (imagepca.get()) {
			SharedDataLock imagepca_lock(imagepca->mutex);
			(*imagepca)->roi = empty;
		}

		if (gradientpca.get()) {
			SharedDataLock gradientpca_lock(gradientpca->mutex);
			(*gradientpca)->roi = empty;
		}
	}

	SharedDataLock image_lock(image->mutex);
	cv::Rect oldRoi = (*image)->roi;
	cv::Rect newRoi = roi;
	image_lock.unlock();

	cv::Rect isecGlob = oldRoi & newRoi;
	cv::Rect isecOld(0, 0, 0, 0);
	cv::Rect isecNew(0, 0, 0, 0);
	if (isecGlob.width > 0 && isecGlob.height > 0) {
		isecOld.x = isecGlob.x - oldRoi.x;
		isecOld.y = isecGlob.y - oldRoi.y;
		isecOld.width = isecGlob.width;
		isecOld.height = isecGlob.height;

		isecNew.x = isecGlob.x - newRoi.x;
		isecNew.y = isecGlob.y - newRoi.y;
		isecNew.width = isecGlob.width;
		isecNew.height = isecGlob.height;
	}

	std::vector<cv::Rect> sub;
	int subArea = MultiImg::Auxiliary::RectComplement(
		oldRoi.width, oldRoi.height, isecOld, sub);

	std::vector<cv::Rect> add;
	int addArea = MultiImg::Auxiliary::RectComplement(
		newRoi.width, newRoi.height, isecNew, add);

	bool profitable = (subArea + addArea) < (newRoi.width * newRoi.height);

	sets_ptr tmp_sets_image(new SharedData<std::vector<BinSet> >(NULL));
	sets_ptr tmp_sets_imagepca(new SharedData<std::vector<BinSet> >(NULL));
	sets_ptr tmp_sets_gradient(new SharedData<std::vector<BinSet> >(NULL));
	sets_ptr tmp_sets_gradientpca(new SharedData<std::vector<BinSet> >(NULL));
	if (reuse && profitable) {
		if (!viewIMG->isPayloadHidden())
			viewIMG->subImage(tmp_sets_image, sub, roi);
		if (!viewGRAD->isPayloadHidden())
			viewGRAD->subImage(tmp_sets_gradient, sub, roi);
		if (imagepca.get())
			viewIMGPCA->subImage(tmp_sets_imagepca, sub, roi);
		if (gradientpca.get())
			viewGRADPCA->subImage(tmp_sets_gradientpca, sub, roi);
	}

	updateRGB(true);
	rgbDock->setEnabled(true);

	labels = cv::Mat1s(full_labels, roi);
	bandView->labels = labels;
	for (size_t i = 0; i < viewers.size(); ++i)
		viewers[i]->labels = labels;

	size_t numbands;
	{
		SharedMultiImgBaseGuard guard(*image_lim);
		numbands = bandsSlider->value();
		if (numbands <= 2)
			numbands = 3;
		if (numbands > (*image_lim)->size())
			numbands = (*image_lim)->size();
		for (size_t i = 0; i < viewers.size(); ++i) {
			if (viewers[i]->getSelection() >= numbands)
				viewers[i]->setSelection(0);
		}
	}

	SharedMultiImgPtr scoped_image(new SharedMultiImgBase(NULL));
	BackgroundTaskPtr taskScope(new MultiImg::ScopeImage(
		image_lim, scoped_image, roi));
	queue.push(taskScope);

	// each vector's size is atmost #bands (e.g., gradient has one less)
	bands.assign(viewers.size(), std::vector<QPixmap*>(numbands, NULL));

	BackgroundTaskPtr taskRescale(new MultiImg::RescaleTbb(
		scoped_image, image, numbands, roi));
	queue.push(taskRescale);

	{
		SharedDataLock hlock(normIMGRange->mutex);
		double min = (*normIMGRange)->first;
		double max = (*normIMGRange)->second;
		hlock.unlock();

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_DATARANGE) {
			BackgroundTaskPtr taskImgNormRange(new NormRangeCuda(
				image, normIMGRange, normIMG, 0, min, max, true, roi));
			queue.push(taskImgNormRange);
		} else {
			BackgroundTaskPtr taskImgNormRange(new NormRangeTbb(
				image, normIMGRange, normIMG, 0, min, max, true, roi));
			queue.push(taskImgNormRange);
		}
	}

	if (!viewIMG->isPayloadHidden()) {
		if (reuse && profitable) {
			viewIMG->addImage(tmp_sets_image, add, roi);	
		} else {
			viewIMG->setImage(image, roi);
		}
	}

	BackgroundTaskPtr taskImgFinish(new BackgroundTask(roi));
	QObject::connect(taskImgFinish.get(), SIGNAL(finished(bool)), 
		this, SLOT(imgCalculationComplete(bool)), Qt::QueuedConnection);
	queue.push(taskImgFinish);

	if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_GRADIENT) {
		BackgroundTaskPtr taskGradient(new MultiImg::GradientCuda(
			image, gradient, roi));
		queue.push(taskGradient);
	} else {
		BackgroundTaskPtr taskGradient(new MultiImg::GradientTbb(
			image, gradient, roi));
		queue.push(taskGradient);
	}

	{
		SharedDataLock hlock(normGRADRange->mutex);
		double min = (*normGRADRange)->first;
		double max = (*normGRADRange)->second;
		hlock.unlock();

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_DATARANGE) {
			BackgroundTaskPtr taskGradNormRange(new NormRangeCuda(
				gradient, normGRADRange, normGRAD, 1, min, max, true, roi));
			queue.push(taskGradNormRange);
		} else {
			BackgroundTaskPtr taskGradNormRange(new NormRangeTbb(
				gradient, normGRADRange, normGRAD, 1, min, max, true, roi));
			queue.push(taskGradNormRange);
		}
	}

	if (!viewGRAD->isPayloadHidden()) {
		if (reuse && profitable) {
			viewGRAD->addImage(tmp_sets_gradient, add, roi);
		} else {
			viewGRAD->setImage(gradient, roi);
		}
	}

	BackgroundTaskPtr taskGradFinish(new BackgroundTask(roi));
	QObject::connect(taskGradFinish.get(), SIGNAL(finished(bool)), 
		this, SLOT(gradCalculationComplete(bool)), Qt::QueuedConnection);
	queue.push(taskGradFinish);

	if (imagepca.get()) {
		BackgroundTaskPtr taskPca(new MultiImg::PcaTbb(
			image, imagepca, 0, roi));
		queue.push(taskPca);

		if (reuse && profitable) {
			viewIMGPCA->addImage(tmp_sets_imagepca, add, roi);
		} else {
			viewIMGPCA->setImage(imagepca, roi);
		}

		BackgroundTaskPtr taskImgPcaFinish(new BackgroundTask(roi));
		QObject::connect(taskImgPcaFinish.get(), SIGNAL(finished(bool)), 
			this, SLOT(imgPcaCalculationComplete(bool)), Qt::QueuedConnection);
		queue.push(taskImgPcaFinish);
	}

	if (gradientpca.get()) {
		BackgroundTaskPtr taskPca(new MultiImg::PcaTbb(
			gradient, gradientpca, 0, roi));
		queue.push(taskPca);

		if (reuse && profitable) {
			viewGRADPCA->addImage(tmp_sets_gradientpca, add, roi);
		} else {
			viewGRADPCA->setImage(gradientpca, roi);
		}

		BackgroundTaskPtr taskGradPcaFinish(new BackgroundTask(roi));
		QObject::connect(taskGradPcaFinish.get(), SIGNAL(finished(bool)), 
			this, SLOT(gradPcaCalculationComplete(bool)), Qt::QueuedConnection);
		queue.push(taskGradPcaFinish);
	}

	BackgroundTaskPtr roiFinish(new BackgroundTask(roi));
	QObject::connect(roiFinish.get(), SIGNAL(finished(bool)), 
		this, SLOT(finishROIChange(bool)), Qt::QueuedConnection);
	queue.push(roiFinish);
}

void MainWindow::imgCalculationComplete(bool success)
{
	if (success) 
		finishViewerRefresh(IMG);
}

void MainWindow::gradCalculationComplete(bool success)
{
	if (success) 
		finishViewerRefresh(GRAD);
}

void MainWindow::imgPcaCalculationComplete(bool success)
{
	if (success) 
		finishViewerRefresh(IMGPCA);
}

void MainWindow::gradPcaCalculationComplete(bool success)
{
	if (success)
		finishViewerRefresh(GRADPCA);
}

void MainWindow::disconnectViewer(int viewer)
{
	disconnect(bandView, SIGNAL(pixelOverlay(int, int)),
		viewers[viewer], SLOT(overlay(int, int)));
	disconnect(bandView, SIGNAL(killHover()),
		viewers[viewer]->getViewport(), SLOT(killHover()));
	disconnect(bandView, SIGNAL(subPixels(const std::map<std::pair<int, int>, short> &)),
		viewers[viewer], SLOT(subPixels(const std::map<std::pair<int, int>, short> &)));
	disconnect(bandView, SIGNAL(addPixels(const std::map<std::pair<int, int>, short> &)),
		viewers[viewer], SLOT(addPixels(const std::map<std::pair<int, int>, short> &)));
}

void MainWindow::finishViewerRefresh(int viewer)
{
	viewers[viewer]->setEnabled(true);
	connect(bandView, SIGNAL(pixelOverlay(int, int)),
		viewers[viewer], SLOT(overlay(int, int)));
	connect(bandView, SIGNAL(killHover()),
		viewers[viewer]->getViewport(), SLOT(killHover()));
	connect(bandView, SIGNAL(subPixels(const std::map<std::pair<int, int>, short> &)),
		viewers[viewer], SLOT(subPixels(const std::map<std::pair<int, int>, short> &)));
	connect(bandView, SIGNAL(addPixels(const std::map<std::pair<int, int>, short> &)),
		viewers[viewer], SLOT(addPixels(const std::map<std::pair<int, int>, short> &)));
	if (viewer == GRAD) {
		normTargetChanged(true);
	}
	if (activeViewer->getType() == viewer) {
		MainWindow::updateBand();
	}
}

void MainWindow::initUI()
{
	/* GUI elements */
	initGraphsegUI();
	initIlluminantUI();
#ifdef WITH_SEG_MEANSHIFT
	initUnsupervisedSegUI();
#endif
	initNormalizationUI();

	/* more manual work to get GUI in proper shape */
	graphsegWidget->hide();

	// start with IMG, hide IMGPCA, GRADPCA at the beginning
	activeViewer = viewIMG;
	viewIMG->setActive();
	viewIMGPCA->toggleFold();
	viewGRADPCA->toggleFold();

	// dock arrangement
	tabifyDockWidget(rgbDock, roiDock);
	tabifyDockWidget(labelDock, illumDock);
	tabifyDockWidget(labelDock, normDock);
#ifdef WITH_SEG_MEANSHIFT
	usDock->hide();
#else
	usDock->deleteLater();
#endif

	/* slots & signals */
	connect(docksButton, SIGNAL(clicked()),
			this, SLOT(openContextMenu()));

	connect(bandDock, SIGNAL(topLevelChanged(bool)),
			this, SLOT(reshapeDock(bool)));

	connect(markerSelector, SIGNAL(currentIndexChanged(int)),
			bandView, SLOT(changeLabel(int)));
	connect(clearButton, SIGNAL(clicked()),
			this, SLOT(labelflush()));

	connect(bandView, SIGNAL(newLabel()),
			this, SLOT(createLabel()));

	connect(ignoreButton, SIGNAL(toggled(bool)),
			this, SLOT(toggleLabels(bool)));

	// label buttons
	connect(lLoadButton, SIGNAL(clicked()),
			this, SLOT(loadLabeling()));
	connect(lSaveButton, SIGNAL(clicked()),
			this, SLOT(saveLabeling()));
	connect(lLoadSeedButton, SIGNAL(clicked()),
			this, SLOT(loadSeeds()));

	// signals for ROI
	connect(roiButtons, SIGNAL(clicked(QAbstractButton*)),
			this, SLOT(ROIDecision(QAbstractButton*)));
	connect(roiView, SIGNAL(newSelection(QRect)),
			this, SLOT(ROISelection(QRect)));

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

	connect(addButton, SIGNAL(clicked()),
			this, SLOT(addToLabel()));
	connect(remButton, SIGNAL(clicked()),
			this, SLOT(remFromLabel()));
	connect(this, SIGNAL(alterLabel(const multi_img::Mask&,bool)),
			bandView, SLOT(alterLabel(const multi_img::Mask&,bool)));
	connect(this, SIGNAL(clearLabel()),
			bandView, SLOT(clearLabelPixels()));
	connect(this, SIGNAL(drawOverlay(const multi_img::Mask&)),
			bandView, SLOT(drawOverlay(const multi_img::Mask&)));
	connect(applyButton, SIGNAL(clicked()),
			bandView, SLOT(updateLabels()));
	connect(&bandView->labelTimer, SIGNAL(timeout()), 
			bandView, SLOT(commitLabelChanges()));
	connect(bandView, SIGNAL(refreshLabels()),
			this, SLOT(refreshLabelsInViewers()));

	connect(this, SIGNAL(newLabelColors(const QVector<QColor>&, bool)),
			bandView, SLOT(setLabelColors(const QVector<QColor>&, bool)));
	connect(alphaSlider, SIGNAL(valueChanged(int)),
			bandView, SLOT(applyLabelAlpha(int)));


	connect(bandView, SIGNAL(killHover()),
			viewerContainer, SIGNAL(viewportsKillHover()));
	connect(bandView, SIGNAL(pixelOverlay(int, int)),
			viewerContainer, SIGNAL(viewersOverlay(int, int)));
	connect(bandView, SIGNAL(subPixels(const std::map<std::pair<int, int>, short> &)),
			viewerContainer, SIGNAL(viewersSubPixels(const std::map<std::pair<int, int>, short> &)));
	connect(bandView, SIGNAL(addPixels(const std::map<std::pair<int, int>, short> &)),
			viewerContainer, SIGNAL(viewersAddPixels(const std::map<std::pair<int, int>, short> &)));
	connect(bandView, SIGNAL(newSingleLabel(short)),
			viewerContainer, SIGNAL(viewersHighlight(short)));

	connect(markButton, SIGNAL(toggled(bool)),
			viewerContainer, SIGNAL(viewersToggleLabeled(bool)));
	connect(nonmarkButton, SIGNAL(toggled(bool)),
			viewerContainer, SIGNAL(viewersToggleUnlabeled(bool)));

//	// for self-activation of viewports
//	QSignalMapper *vpmap = new QSignalMapper(this);
//	for (size_t i = 0; i < viewers.size(); ++i)
//		vpmap->setMapping(viewers[i]->getViewport(), (int)i);
//	connect(vpmap, SIGNAL(mapped(int)),
//			this, SLOT(setActive(int)));

//	for (size_t i = 0; i < viewers.size(); ++i)
//	{
//		multi_img_viewer *v = viewers[i];
//		const Viewport *vp = v->getViewport();

//		connect(markButton, SIGNAL(toggled(bool)),
//				v, SLOT(toggleLabeled(bool)));
//		connect(nonmarkButton, SIGNAL(toggled(bool)),
//				v, SLOT(toggleUnlabeled(bool)));

//		if (!v->isPayloadHidden()) {
//			connect(bandView, SIGNAL(pixelOverlay(int, int)),
//					v, SLOT(overlay(int, int)));
//			connect(bandView, SIGNAL(killHover()),
//					vp, SLOT(killHover()));
//			connect(bandView, SIGNAL(subPixels(const std::map<std::pair<int, int>, short> &)),
//					v, SLOT(subPixels(const std::map<std::pair<int, int>, short> &)));
//			connect(bandView, SIGNAL(addPixels(const std::map<std::pair<int, int>, short> &)),
//					v, SLOT(addPixels(const std::map<std::pair<int, int>, short> &)));
//		}

//		connect(vp, SIGNAL(activated()),
//				vpmap, SLOT(map()));

//		for (size_t j = 0; j < viewers.size(); ++j) {
//			multi_img_viewer *v2 = viewers[j];
//			const Viewport *vp2 = v2->getViewport();
//			// connect folding signal to all viewports
//			connect(v, SIGNAL(folding()),
//					vp2, SLOT(folding()));

//			if (i == j)
//				continue;
//			// connect activation signal to all *other* viewers
//			connect(vp, SIGNAL(activated()),
//			        v2, SLOT(setInactive()));
//		}

//		connect(vp, SIGNAL(bandSelected(representation, int)),
//				this, SLOT(selectBand(representation, int)));
	connect(viewerContainer, SIGNAL(viewPortBandSelected(representation,int)),
			this, SLOT(selectBand(representation,int));

		// TODO ViewerContainer
//		connect(v, SIGNAL(setGUIEnabled(bool, TaskType)),
//				this, SLOT(setGUIEnabled(bool, TaskType)));

	connect(viewerContainer, SIGNAL(viewerSetGUIEnabled(bool,TaskType)),
			this, SLOT(setGUIEnabled(bool,TaskType)));

		// TODO ViewerContainer
//		connect(v, SIGNAL(toggleViewer(bool , representation)),
//				this, SLOT(toggleViewer(bool , representation)));

		// TODO ViewerContainer
//		connect(v, SIGNAL(finishTask(bool)),
//				this, SLOT(finishTask(bool)));

		// TODO ViewerContainer
//		connect(v, SIGNAL(newOverlay()),
//				this, SLOT(newOverlay()));
		// TODO ViewerContainer
//		connect(vp, SIGNAL(newOverlay(int)),
//				this, SLOT(newOverlay()));

		// TODO ViewerContainer
//		connect(vp, SIGNAL(addSelection()),
//				this, SLOT(addToLabel()));

		// TODO ViewerContainer
//		connect(vp, SIGNAL(remSelection()),
//				this, SLOT(remFromLabel()));
		// TODO ViewerContainer
//		connect(bandView, SIGNAL(newSingleLabel(short)),
//				vp, SLOT(highlight(short)));
//	}

	/// init bandsSlider
	bandsLabel->setText(QString("%1 bands").arg((*image_lim)->size()));
	bandsSlider->setMaximum((*image_lim)->size());
	bandsSlider->setValue((*image_lim)->size());
	connect(bandsSlider, SIGNAL(valueChanged(int)),
			this, SLOT(bandsSliderMoved(int)));
	connect(bandsSlider, SIGNAL(sliderMoved(int)),
			this, SLOT(bandsSliderMoved(int)));

	/// global shortcuts
	QShortcut *scr = new QShortcut(Qt::CTRL + Qt::Key_S, this);
	connect(scr, SIGNAL(activated()), this, SLOT(screenshot()));

	BackgroundTaskPtr taskRgb(new RgbTbb(
		image_lim, mat3f_ptr(new SharedData<cv::Mat3f>(new cv::Mat3f)), full_rgb_temp));
	taskRgb->run();
	updateRGB(true);
}

void MainWindow::toggleViewer(bool enable, representation viewer)
{
	if (!enable) {
		disconnectViewer(viewer);

		if (viewer == IMG) {
			// no-op
		} else if (viewer == GRAD) {
			// no-op
		} else if (viewer == IMGPCA) {
			viewers[IMGPCA]->resetImage();
			imagepca.reset();
			if (viewers[IMGPCA] == activeViewer) {
				viewers[IMG]->activateViewport();
				updateBand();
			}
		} else if (viewer == GRADPCA) {
			viewers[GRADPCA]->resetImage();
			gradientpca.reset();
			if (viewers[GRADPCA] == activeViewer) {
				viewers[IMG]->activateViewport();
				updateBand();
			}
		}
	} else {
		setGUIEnabled(false, TT_TOGGLE_VIEWER);
		viewers[viewer]->setEnabled(false);

		if (viewer == IMG) {
			viewers[viewer]->setImage(image, roi);

			BackgroundTaskPtr taskImgFinish(new BackgroundTask(roi));
			QObject::connect(taskImgFinish.get(), SIGNAL(finished(bool)), 
				this, SLOT(imgCalculationComplete(bool)), Qt::QueuedConnection);
			queue.push(taskImgFinish);
		} else if (viewer == GRAD) {
			viewers[viewer]->setImage(gradient, roi);

			BackgroundTaskPtr taskGradFinish(new BackgroundTask(roi));
			QObject::connect(taskGradFinish.get(), SIGNAL(finished(bool)), 
				this, SLOT(gradCalculationComplete(bool)), Qt::QueuedConnection);
			queue.push(taskGradFinish);
		} else if (viewer == IMGPCA) {
			imagepca.reset(new SharedMultiImgBase(new multi_img(0, 0, 0)));

			BackgroundTaskPtr taskPca(new MultiImg::PcaTbb(
				image, imagepca, 0, roi));
			queue.push(taskPca);

			viewers[viewer]->setImage(imagepca, roi);

			BackgroundTaskPtr taskImgPcaFinish(new BackgroundTask(roi));
			QObject::connect(taskImgPcaFinish.get(), SIGNAL(finished(bool)), 
				this, SLOT(imgPcaCalculationComplete(bool)), Qt::QueuedConnection);
			queue.push(taskImgPcaFinish);
		} else if (viewer == GRADPCA) {
			gradientpca.reset(new SharedMultiImgBase(new multi_img(0, 0, 0)));

			BackgroundTaskPtr taskPca(new MultiImg::PcaTbb(
				gradient, gradientpca, 0, roi));
			queue.push(taskPca);

			viewers[viewer]->setImage(gradientpca, roi);

			BackgroundTaskPtr taskGradPcaFinish(new BackgroundTask(roi));
			QObject::connect(taskGradPcaFinish.get(), SIGNAL(finished(bool)), 
				this, SLOT(gradPcaCalculationComplete(bool)), Qt::QueuedConnection);
			queue.push(taskGradPcaFinish);
		} 

		BackgroundTaskPtr taskEpilog(new BackgroundTask(roi));
		QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
			this, SLOT(finishTask(bool)), Qt::QueuedConnection);
		queue.push(taskEpilog);
	}
}

void MainWindow::setGUIEnabled(bool enable, TaskType tt)
{
	bandsSlider->setEnabled(enable || tt == TT_BAND_COUNT);
	ignoreButton->setEnabled(enable || tt == TT_TOGGLE_LABELS);
	addButton->setEnabled(enable);
	remButton->setEnabled(enable);

	viewerContainer->setGUIEnabled(enable, tt);
	// -> now in ViewerContainer
//	viewIMG->enableBinSlider(enable);
//	viewIMG->setEnabled(enable || tt == TT_BIN_COUNT || tt == TT_TOGGLE_VIEWER);
//	viewGRAD->enableBinSlider(enable);
//	viewGRAD->setEnabled(enable || tt == TT_BIN_COUNT || tt == TT_TOGGLE_VIEWER);
//	viewIMGPCA->enableBinSlider(enable);
//	viewIMGPCA->setEnabled(enable || tt == TT_BIN_COUNT || tt == TT_TOGGLE_VIEWER);
//	viewGRADPCA->enableBinSlider(enable);
//	viewGRADPCA->setEnabled(enable || tt == TT_BIN_COUNT || tt == TT_TOGGLE_VIEWER);

	applyButton->setEnabled(enable);
	clearButton->setEnabled(enable);
	bandView->setEnabled(enable);
	graphsegWidget->setEnabled(enable);

	normDock->setEnabled((enable || tt == TT_NORM_RANGE || tt == TT_CLAMP_RANGE_IMG || tt == TT_CLAMP_RANGE_GRAD) && !limitedMode);
	normIButton->setEnabled(enable || tt == TT_NORM_RANGE || tt == TT_CLAMP_RANGE_IMG);
	normGButton->setEnabled(enable || tt == TT_NORM_RANGE || tt == TT_CLAMP_RANGE_GRAD);
	normModeBox->setEnabled(enable);
	normApplyButton->setEnabled(enable || tt == TT_NORM_RANGE);
	normClampButton->setEnabled(enable || tt == TT_CLAMP_RANGE_IMG || tt == TT_CLAMP_RANGE_GRAD);

	labelDock->setEnabled(enable);

	rgbDock->setEnabled(enable);

	illumDock->setEnabled((enable || tt == TT_APPLY_ILLUM) && !limitedMode);

	usDock->setEnabled(enable && !limitedMode);

	roiDock->setEnabled(enable || tt == TT_SELECT_ROI);

	runningTask = tt;
}

void MainWindow::bandsSliderMoved(int b)
{
	bandsLabel->setText(QString("%1 bands").arg(b));
	if (!bandsSlider->isSliderDown()) {
		queue.cancelTasks(roi);
		
		setGUIEnabled(false, TT_BAND_COUNT);

		applyROI(false);

		BackgroundTaskPtr taskEpilog(new BackgroundTask(roi));
		QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
			this, SLOT(finishTask(bool)), Qt::QueuedConnection);
		queue.push(taskEpilog);
	}
}

void MainWindow::toggleLabels(bool toggle)
{
	queue.cancelTasks();
	setGUIEnabled(false, TT_TOGGLE_LABELS);

	for (size_t i = 0; i < viewers.size(); ++i)
		viewers[i]->toggleLabels(toggle);

	BackgroundTaskPtr taskEpilog(new BackgroundTask());
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
		this, SLOT(finishTask(bool)), Qt::QueuedConnection);
	queue.push(taskEpilog);
}

#ifdef WITH_SEG_MEANSHIFT
void MainWindow::usMethodChanged(int idx)
{
	// idx: 0 Meanshift, 1 Medianshift, 2 Probabilistic Shift
	usSkipPropWidget->setEnabled(idx == 1);
	usSpectralWidget->setEnabled(idx == 2);
	usMSPPWidget->setEnabled(idx == 2);
}

void MainWindow::usInitMethodChanged(int idx)
{
	switch (usInitMethodBox->itemData(idx).toInt()) {
	case vole::JUMP:
		usInitPercentWidget->hide();
		usInitJumpWidget->show();
		break;
	case vole::PERCENT:
		usInitJumpWidget->hide();
		usInitPercentWidget->show();
		break;
	default:
		usInitJumpWidget->hide();
		usInitPercentWidget->hide();
	}
}
#endif

bool MainWindow::setLabelColors(const std::vector<cv::Vec3b> &colors)
{
	QVector<QColor> col = vole::Vec2QColor(colors);
	col[0] = Qt::white; // override black for 0 label

	// test if costy rebuilds necessary (existing colors changed)
	bool changed = false;
	for (int i = 1; i < labelColors.size() && i < col.size(); ++i) {
		if (col[i] != labelColors[i])
			changed = true;
	}

	labelColors = col;

	// use colors for our awesome label menu
	markerSelector->clear();
	for (int i = 1; i < labelColors.size(); ++i) // 0 is index for unlabeled
	{
		markerSelector->addItem(colorIcon(labelColors[i]), "");
	}
	markerSelector->addItem(QIcon(":/toolbar/add"), "");

	// tell others about colors
	emit newLabelColors(labelColors, changed);

	if (changed) 
		setGUIEnabled(false);

	for (size_t i = 0; i < viewers.size(); ++i)
		viewers[i]->updateLabelColors(labelColors, changed);

	if (changed) {
		BackgroundTaskPtr taskEpilog(new BackgroundTask());
		QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
			this, SLOT(finishTask(bool)), Qt::QueuedConnection);
		queue.push(taskEpilog);
	}

	return changed;
}

void MainWindow::setLabels(const vole::Labeling &labeling)
{
	SharedDataLock image_lock(image->mutex);
	assert(labeling().rows == (*image)->height && labeling().cols == (*image)->width);
	image_lock.unlock();

	/* note: always update labels before updating label colors, for the case
	   that there are less colors available than used in previous labeling */
	cv::Mat1s labels = labeling();
	// following assignments are probably redundant (OpenCV shallow copy)
	bandView->labels = labels;
	for (size_t i = 0; i < viewers.size(); ++i)
		viewers[i]->labels = labels;

	bool updated = setLabelColors(labeling.colors());
	if (!updated) {
		bandView->refresh();
		refreshLabelsInViewers();
	}
}

void MainWindow::refreshLabelsInViewers()
{
	setGUIEnabled(false);
	for (size_t i = 0; i < viewers.size(); ++i)
		viewers[i]->updateLabels();

	BackgroundTaskPtr taskEpilog(new BackgroundTask());
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
		this, SLOT(finishTask(bool)), Qt::QueuedConnection);
	queue.push(taskEpilog);
}

void MainWindow::createLabel()
{
	int index = labelColors.count();
	// increment colors by 1
	setLabelColors(vole::Labeling::colors(index + 1, true));
	// select our new label for convenience
	markerSelector->setCurrentIndex(index - 1);
}

const QPixmap* MainWindow::getBand(representation type, int dim)
{
	std::vector<QPixmap*> &v = bands[type];

	if (!v[dim]) {
		SharedMultiImgPtr multi = viewers[type]->getImage();
		qimage_ptr qimg(new SharedData<QImage>(new QImage()));

		SharedDataLock hlock(multi->mutex);

		BackgroundTaskPtr taskConvert(new MultiImg::Band2QImageTbb(multi, qimg, dim));
		taskConvert->run();

		hlock.unlock();

		v[dim] = new QPixmap(QPixmap::fromImage(**qimg));
	}
	return v[dim];subImage
}

void MainWindow::updateBand()
{
	selectBand(activeViewer->getType(),
			   activeViewer->getSelection());
	bandView->update();
}

void MainWindow::selectBand(representation type, int dim)
{
	bandView->setEnabled(true);
	bandView->setPixmap(*getBand(type, dim));
	SharedMultiImgPtr m = viewers[type]->getImage();
	SharedDataLock hlock(m->mutex);
	std::string banddesc = (*m)->meta[dim].str();
	hlock.unlock();
	QString title;
	if (banddesc.empty())
		title = QString("%1 Band #%2")
			.arg(type == GRAD ? "Gradient" : "Image") // TODO
			.arg(dim+1);
	else
		title = QString("%1 Band %2")
			.arg(type == GRAD ? "Gradient" : "Image") // TODO
			.arg(banddesc.c_str());

	bandDock->setWindowTitle(title);
}

void MainWindow::applyIlluminant() {
	int i1 = i1Box->itemData(i1Box->currentIndex()).value<int>();
	int i2 = i2Box->itemData(i2Box->currentIndex()).value<int>();
	if (i1 == i2)
		return;

	i1Box->setDisabled(true);
	i1Check->setVisible(true);

	queue.cancelTasks(roi);
	setGUIEnabled(false, TT_APPLY_ILLUM);

	/* remove old illuminant */
	if (i1 != 0) {
		const Illuminant &il = getIlluminant(i1);

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_ILLUMINANT) {
			BackgroundTaskPtr taskIllum(new MultiImg::IlluminantCuda(
				image_lim, il, true, roi, false));
			queue.push(taskIllum);
		} else {
			BackgroundTaskPtr taskIllum(new MultiImg::IlluminantTbb(
				image_lim, il, true, roi, false));
			queue.push(taskIllum);
		}
	}

	/* add new illuminant */
	if (i2 != 0) {
		const Illuminant &il = getIlluminant(i2);

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_ILLUMINANT) {
			BackgroundTaskPtr taskIllum(new MultiImg::IlluminantCuda(
				image_lim, il, false, roi, false));
			queue.push(taskIllum);
		} else {
			BackgroundTaskPtr taskIllum(new MultiImg::IlluminantTbb(
				image_lim, il, false, roi, false));
			queue.push(taskIllum);
		}
	}

	std::vector<multi_img::Value> empty;
	viewIMG->setIlluminant((i2 ? getIlluminantC(i2) : empty), true);

	applyROI(false);
	rgbDock->setEnabled(false);

	BackgroundTaskPtr taskRgb(new RgbTbb(
		image_lim, mat3f_ptr(new SharedData<cv::Mat3f>(new cv::Mat3f)), full_rgb_temp, roi));
	QObject::connect(taskRgb.get(), SIGNAL(finished(bool)), this, SLOT(updateRGB(bool)), Qt::QueuedConnection);
	queue.push(taskRgb);

	BackgroundTaskPtr taskEpilog(new BackgroundTask(roi));
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
		this, SLOT(finishTask(bool)), Qt::QueuedConnection);
	queue.push(taskEpilog);

	/* reflect change in our own gui (will propagate to viewIMG) */
	i1Box->setCurrentIndex(i2Box->currentIndex());
}

void MainWindow::updateRGB(bool success)
{
	if (!success)
		return;

	SharedDataLock hlock(full_rgb_temp->mutex);
	if (!(*full_rgb_temp)->isNull()) {
		full_rgb = QPixmap::fromImage(**full_rgb_temp);
	}
	hlock.unlock();

	roiView->setPixmap(full_rgb);
	roiView->update();
	if (roi.width > 0) {
		rgb = full_rgb.copy(roi.x, roi.y, roi.width, roi.height);
		rgbView->setPixmap(rgb);
		rgbView->update();
	}
}

void MainWindow::initIlluminantUI()
{
	for (int i = 0; i < 2; ++i) {
		QComboBox *b = (i ? i2Box : i1Box);
		b->addItem("Neutral", 0);
		b->addItem("2,856 K (Illuminant A, light bulb)",	2856);
		b->addItem("3,100 K (Tungsten halogen lamp)",		3100);
		b->addItem("5,000 K (Horizon light)",				5000);
		b->addItem("5,500 K (Mid-morning daylight)",		5500);
		b->addItem("6,500 K (Noon daylight)",				6500);
		b->addItem("7,500 K (North sky daylight)",			7500);
	}
	connect(i2Button, SIGNAL(clicked()),
			this, SLOT(applyIlluminant()));
	connect(i1Box, SIGNAL(currentIndexChanged(int)),
			this, SLOT(setI1(int)));
	connect(i1Check, SIGNAL(toggled(bool)),
			this, SLOT(setI1Visible(bool)));
	i1Check->setVisible(false);
}
subImage
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
	/* reset gui to current settings */
	int target = (normIButton->isChecked() ? 0 : 1);
	MultiImg::NormMode m = (target == 0 ? normIMG : normGRAD);

	// update normModeBox
	normModeBox->setCurrentIndex(m);

	// update norm range spin boxes
	normModeSelected(m, true, usecurrent);
}

void MainWindow::normModeSelected(int mode, bool targetchange, bool usecurrent)
{
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
}

void MainWindow::normModeFixed()
{
	if (normModeBox->currentIndex() != 2)
		normModeBox->setCurrentIndex(2);
}

void MainWindow::applyNormUserRange()
{
	int target = (normIButton->isChecked() ? 0 : 1);

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
		viewIMG->updateBinning(-1);

		BackgroundTaskPtr taskFinishNorm(new BackgroundTask());
		QObject::connect(taskFinishNorm.get(), SIGNAL(finished(bool)), 
			this, SLOT(finishNormRangeImgChange(bool)), Qt::QueuedConnection);
		queue.push(taskFinishNorm);
	} else {
		viewGRAD->updateBinning(-1);

		BackgroundTaskPtr taskFinishNorm(new BackgroundTask());
		QObject::connect(taskFinishNorm.get(), SIGNAL(finished(bool)), 
			this, SLOT(finishNormRangeGradChange(bool)), Qt::QueuedConnection);
		queue.push(taskFinishNorm);
	}

	BackgroundTaskPtr taskEpilog(new BackgroundTask());
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
		this, SLOT(finishTask(bool)), Qt::QueuedConnection);
	queue.push(taskEpilog);
}

void MainWindow::finishNormRangeImgChange(bool success)
{
	if (success) {
		SharedDataLock hlock(image->mutex);
		bands[GRAD].assign((*image)->size(), NULL);
		hlock.unlock();
		updateBand();
	}
}

void MainWindow::finishNormRangeGradChange(bool success)
{
	if (success) {
		SharedDataLock hlock(gradient->mutex);
		bands[GRAD].assign((*gradient)->size(), NULL);
		hlock.unlock();
		updateBand();
	}
}

void MainWindow::clampNormUserRange()
{
	int target = (normIButton->isChecked() ? 0 : 1);

	// set internal norm mode
	MultiImg::NormMode &nm = (target == 0 ? normIMG : normGRAD);
	nm = static_cast<MultiImg::NormMode>(normModeBox->currentIndex());

	/* if image is changed, change full image. for gradient, we cannot preserve
		the gradient over ROI or illuminant changes, so it remains a local change */
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

		BackgroundTaskPtr taskRgb(new RgbTbb(
			image_lim, mat3f_ptr(new SharedData<cv::Mat3f>(new cv::Mat3f)), full_rgb_temp, roi));
		QObject::connect(taskRgb.get(), SIGNAL(finished(bool)), this, SLOT(updateRGB(bool)), Qt::QueuedConnection);
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

		viewGRAD->updateBinning(-1);

		BackgroundTaskPtr taskFinishClamp(new BackgroundTask());
		QObject::connect(taskFinishClamp.get(), SIGNAL(finished(bool)), 
			this, SLOT(finishNormRangeGradChange(bool)), Qt::QueuedConnection);
		queue.push(taskFinishClamp);

		BackgroundTaskPtr taskEpilog(new BackgroundTask());
		QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
			this, SLOT(finishTask(bool)), Qt::QueuedConnection);
		queue.push(taskEpilog);
	}
}

void MainWindow::initGraphsegUI()
{
	graphsegSourceBox->addItem("Image", 0);
	graphsegSourceBox->addItem("Gradient", 1); // TODO PCA
	graphsegSourceBox->addItem("Shown Band", 2);
	graphsegSourceBox->setCurrentIndex(0);

	graphsegSimilarityBox->addItem("Manhattan distance (L1)", vole::MANHATTAN);
	graphsegSimilarityBox->addItem("Euclidean distance (L2)", vole::EUCLIDEAN);
	graphsegSimilarityBox->addItem(QString::fromUtf8("Chebyshev distance (L∞)"),
								   vole::CHEBYSHEV);
	graphsegSimilarityBox->addItem("Spectral Angle", vole::MOD_SPEC_ANGLE);
	graphsegSimilarityBox->addItem("Spectral Information Divergence",
								   vole::SPEC_INF_DIV);
	graphsegSimilarityBox->addItem("SID+SAM I", vole::SIDSAM1);
	graphsegSimilarityBox->addItem("SID+SAM II", vole::SIDSAM2);
	graphsegSimilarityBox->addItem("Normalized L2", vole::NORM_L2);
	graphsegSimilarityBox->setCurrentIndex(3);

	graphsegAlgoBox->addItem("Kruskal", vole::KRUSKAL);
	graphsegAlgoBox->addItem("Prim", vole::PRIM);
	graphsegAlgoBox->addItem("Power Watershed q=2", vole::WATERSHED2);
	graphsegAlgoBox->setCurrentIndex(1);

	connect(graphsegButton, SIGNAL(toggled(bool)),
			graphsegWidget, SLOT(setVisible(bool)));
	connect(graphsegButton, SIGNAL(toggled(bool)),
			bandView, SLOT(toggleSeedMode(bool)));
	connect(graphsegGoButton, SIGNAL(clicked()),
			this, SLOT(startGraphseg()));
	connect(this, SIGNAL(seedingDone(bool)),
			graphsegButton, SLOT(setChecked(bool)));
}

void MainWindow::runGraphseg(SharedMultiImgPtr input,
							   const vole::GraphSegConfig &config)
{
	setGUIEnabled(false);
	BackgroundTaskPtr taskGraphseg(new GraphsegBackground(
		config, input, bandView->seedMap, graphsegResult));
	QObject::connect(taskGraphseg.get(), SIGNAL(finished(bool)), 
		this, SLOT(finishGraphSeg(bool)), Qt::QueuedConnection);
	queue.push(taskGraphseg);
}

void MainWindow::finishGraphSeg(bool success)
{
	if (success) {
		/* add segmentation to current labeling */
		emit alterLabel(*(graphsegResult.get()), false);
		refreshLabelsInViewers();

		emit seedingDone();
	}
}

void MainWindow::startGraphseg()
{
	vole::GraphSegConfig conf("graphseg");
	conf.algo = (vole::graphsegalg)
				graphsegAlgoBox->itemData(graphsegAlgoBox->currentIndex())
				.value<int>();
	conf.similarity.measure = (vole::similarity_fun)
	      graphsegSimilarityBox->itemData(graphsegSimilarityBox->currentIndex())
	      .value<int>();
#ifdef WITH_EDGE_DETECT
	conf.som_similarity = false;
#endif
	conf.geodesic = graphsegGeodCheck->isChecked();
	conf.multi_seed = false;
	int src = graphsegSourceBox->itemData(graphsegSourceBox->currentIndex())
								 .value<int>();
	if (src == 0) {
		runGraphseg(image, conf);
	} else if (src == 1) {
		runGraphseg(gradient, conf);
	} else {	// currently shown band, construct from selection in viewport
		int band = activeViewer->getSelection();
		SharedMultiImgPtr img = activeViewer->getImage();
		SharedDataLock img_lock(img->mutex);
		SharedMultiImgPtr i(new SharedMultiImgBase(
			new multi_img((**img)[band], (*img)->minval, (*img)->maxval)));
		img_lock.unlock();
		runGraphseg(i, conf);
	}
}

#ifdef WITH_SEG_MEANSHIFT
void MainWindow::initUnsupervisedSegUI()
{
	usMethodBox->addItem("Meanshift", 0);
#ifdef WITH_SEG_MEDIANSHIFT
	usMethodBox->addItem("Medianshift", 1);
#endif
#ifdef WITH_SEG_PROBSHIFT
	usMethodBox->addItem("Probabilistic Shift", 2);
#endif
	usMethodChanged(0); // set default state

	usInitMethodBox->addItem("all", vole::ALL);
	usInitMethodBox->addItem("jump", vole::JUMP);
	usInitMethodBox->addItem("percent", vole::PERCENT);
	usInitMethodChanged(0);

	usBandwidthBox->addItem("adaptive");
	usBandwidthBox->addItem("fixed");
	usBandwidthMethodChanged("adaptive");

	{
		SharedMultiImgBaseGuard guard(*image_lim);
		usBandsSpinBox->setValue((*image_lim)->size());
		usBandsSpinBox->setMaximum((*image_lim)->size());
	}

	// we do not expose the density estimation functionality
	usInitWidget->hide();
	// we also do not expose options exclusive to unavailable methods
#ifndef WITH_SEG_MEDIANSHIFT
	usSkipPropWidget->hide();
#endif
#ifndef WITH_SEG_PROBSHIFT
	usSpectralWidget->hide();
	usMSPPWidget->hide();
#endif

	usInitJumpWidget->hide();
	usInitPercentWidget->hide();
	usFoundKLWidget->hide();
	usProgressWidget->hide();

	connect(usGoButton, SIGNAL(clicked()),
			this, SLOT(startUnsupervisedSeg()));
	connect(usFindKLGoButton, SIGNAL(clicked()),
			this, SLOT(startFindKL()));
	connect(usCancelButton, SIGNAL(clicked()),
			this, SLOT(unsupervisedSegCancelled()));

	connect(usMethodBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(usMethodChanged(int)));

	connect(usLshCheckBox, SIGNAL(toggled(bool)),
			usLshWidget, SLOT(setEnabled(bool)));

	connect(usBandwidthBox, SIGNAL(currentIndexChanged(const QString&)),
			this, SLOT(usBandwidthMethodChanged(const QString&)));

	connect(usInitMethodBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(usInitMethodChanged(int)));

	connect(usSpectralCheckBox, SIGNAL(toggled(bool)),
			usSpectralConvCheckBox, SLOT(setEnabled(bool)));
	connect(usSpectralCheckBox, SIGNAL(toggled(bool)),
			usSpectralMinMaxWidget, SLOT(setEnabled(bool)));

	/// pull default values from temporary instance of config class
	vole::MeanShiftConfig def;
	usKSpinBox->setValue(def.K);
	usLSpinBox->setValue(def.L);
	/// TODO: random seed box
	usPilotKSpinBox->setValue(def.k);
	usInitMethodBox->setCurrentIndex(
			usInitMethodBox->findData(def.starting));
	usInitJumpBox->setValue(def.jump);
	usFixedBWSpinBox->setValue(def.bandwidth);
	usFindKLKMinBox->setValue(def.Kmin);
	usFindKLKStepBox->setValue(def.Kjump);
	usFindKLEpsilonBox->setValue(def.epsilon);

#ifdef WITH_SEG_PROBSHIFT
	vole::ProbShiftConfig def_ps;
	usProbShiftMSPPAlphaSpinBox->setValue(def_ps.msBwFactor);
#endif
}

void MainWindow::usBandwidthMethodChanged(const QString &current) {
	if (current == "fixed") {
		usAdaptiveBWWidget->hide();
		usFixedBWWidget->show();
	} else if (current == "adaptive") {
		usFixedBWWidget->hide();
		usAdaptiveBWWidget->show();
	} else {
		assert(0);
	}
}

void MainWindow::unsupervisedSegCancelled() {
	usCancelButton->setDisabled(true);
	usCancelButton->setText("Please wait...");
	/// runner->terminate() will be called by the Cancel button
}

void MainWindow::startFindKL()
{
	startUnsupervisedSeg(true);
}

void MainWindow::startUnsupervisedSeg(bool findKL)
{
	// allow only one runner at a time (UI enforces that)
	assert(usRunner == NULL);
	usRunner = new CommandRunner();

	int method = usMethodBox->itemData(usMethodBox->currentIndex()).value<int>();

	if (findKL) { // run MeanShift::findKL()
		usRunner->cmd = new vole::MeanShiftShell();
		vole::MeanShiftConfig &config = ((vole::MeanShiftShell *) usRunner->cmd)->config;

		config.batch = true;
		config.findKL = true;
		config.k = usPilotKSpinBox->value();
		config.K = usFindKLKmaxBox->value();
		config.L = usFindKLLmaxBox->value();
		config.Kmin = usFindKLKMinBox->value();
		config.Kjump = usFindKLKStepBox->value();
		config.epsilon = usFindKLEpsilonBox->value();
	} else if (method == 0) { // Meanshift
		usRunner->cmd = new vole::MeanShiftShell();
		vole::MeanShiftConfig &config = ((vole::MeanShiftShell *) usRunner->cmd)->config;

		// fixed settings
		config.batch = true;

		config.use_LSH = usLshCheckBox->isChecked();
		config.K = usKSpinBox->value();
		config.L = usLSpinBox->value();

		config.starting = (vole::ms_sampling) usInitMethodBox->itemData(usInitMethodBox->currentIndex()).value<int>();
		config.percent = usInitPercentBox->value();
		config.jump = usInitJumpBox->value();
		config.k = usPilotKSpinBox->value();

		if (usBandwidthBox->currentText() == "fixed") {
			config.bandwidth = usFixedBWSpinBox->value();
		} else {
			config.bandwidth = 0;
		}
#ifdef WITH_SEG_MEDIANSHIFT
	} else if (method == 1) { // Medianshift
		usRunner->cmd = new vole::MedianShiftShell();
		vole::MedianShiftConfig &config = ((vole::MedianShiftShell *) usRunner->cmd)->config;

		config.K = usKSpinBox->value();
		config.L = usLSpinBox->value();
		config.k = usPilotKSpinBox->value();
		config.skipprop = usSkipPropCheckBox->isChecked();
#endif
#ifdef WITH_SEG_PROBSHIFT
	} else { // Probabilistic Shift
		usRunner->cmd = new vole::ProbShiftShell();
		vole::ProbShiftConfig &config = ((vole::ProbShiftShell *) usRunner->cmd)->config;

		config.useLSH = usLshCheckBox->isChecked();
		config.lshK = usKSpinBox->value();
		config.lshL = usLSpinBox->value();

		config.useSpectral = usSpectralCheckBox->isChecked();
		config.useConverged = usSpectralConvCheckBox->isChecked();
		config.minClusts = usSpectralMinBox->value();
		config.maxClusts = usSpectralMaxBox->value();
		config.useMeanShift = usProbShiftMSPPCheckBox->isChecked();
		config.msBwFactor = usProbShiftMSPPAlphaSpinBox->value();
#endif
	}

	// connect runner with progress bar, cancel button and finish-slot
	connect(usRunner, SIGNAL(progressChanged(int)), usProgressBar, SLOT(setValue(int)));
	connect(usCancelButton, SIGNAL(clicked()), usRunner, SLOT(terminate()));

	qRegisterMetaType< std::map<std::string, boost::any> >("std::map<std::string, boost::any>");
	connect(usRunner, SIGNAL(success(std::map<std::string,boost::any>)), this, SLOT(segmentationApply(std::map<std::string,boost::any>)));
	connect(usRunner, SIGNAL(finished()), this, SLOT(segmentationFinished()));

	usProgressWidget->show();
	usSettingsWidget->setDisabled(true);

	// prepare input image
	boost::shared_ptr<multi_img> input;
	{
		SharedMultiImgBaseGuard guard(*image_lim);
		assert(0 != &**image_lim);
		// FIXME 2013-04-11 georg altmann:
		// not sure what this code is really doing, but this looks like a problem:
		// is input sharing image data with image_lim?
		// If so, another thread could overwrite data while image segmentation is working on it,
		// since there is no locking (unless multi_img does implicit copy on write?).
		input = boost::shared_ptr<multi_img>(
					new multi_img(**image_lim, roi)); // image data is not copied
	}
	int numbands = usBandsSpinBox->value();
	bool gradient = usGradientCheckBox->isChecked();

	if (numbands > 0 && numbands < (int) input->size()) {
		boost::shared_ptr<multi_img> input_tmp(new multi_img(input->spec_rescale(numbands)));
		input = input_tmp;
	}

	if (gradient) {
		// copy needed here
		multi_img loginput(*input);
		loginput.apply_logarithm();
		input = boost::shared_ptr<multi_img>(new multi_img(loginput.spec_gradient()));
	}

	usRunner->input["multi_img"] = input;

	usRunner->start();
}

void MainWindow::segmentationFinished() {
	if (usRunner->abort) {
		// restore Cancel button
		usCancelButton->setEnabled(true);
		usCancelButton->setText("Cancel");
	}

	// hide progress, re-enable settings
	usProgressWidget->hide();
	usSettingsWidget->setEnabled(true);

	/// clean up runner
	delete usRunner;
	usRunner = NULL;
}

void MainWindow::segmentationApply(std::map<std::string, boost::any> output) {
	if (output.count("labels")) {
		boost::shared_ptr<cv::Mat1s> labelMask = boost::any_cast< boost::shared_ptr<cv::Mat1s> >(output["labels"]);
		setLabels(*labelMask);
	}

	if (output.count("findKL.K") && output.count("findKL.L")) {
		int foundK = boost::any_cast<int>(output["findKL.K"]);
		int foundL = boost::any_cast<int>(output["findKL.L"]);
		usFoundKLLabel->setText(QString("Found values: K=%1 L=%2").arg(foundK).arg(foundL));
		usFoundKLWidget->show();
	}
}
#else // method stubs as using define in header does not work (moc problem?)
void MainWindow::startUnsupervisedSeg(bool findKL) {}
void MainWindow::startFindKL() {}
void MainWindow::segmentationFinished() {}
void MainWindow::segmentationApply(std::map<std::string, boost::any>) {}
void MainWindow::usMethodChanged(int idx) {}
void MainWindow::usInitMethodChanged(int idx) {}
void MainWindow::usBandwidthMethodChanged(const QString &current) {}
void MainWindow::unsupervisedSegCancelled() {}
#endif // WITH_SEG_MEANSHIFT

void MainWindow::setActive(int id)
{
	if (viewers[id]->getImage().get()) {
		activeViewer = viewers[id];
	} else {
		activeViewer = viewers[IMG];
	}
}

void MainWindow::labelflush()
{
	std::vector<sets_ptr> tmp_sets;
	cv::Mat1b mask(labels.rows, labels.cols);
	mask = (labels == bandView->getCurLabel());
	bool profitable = ((2 * cv::countNonZero(mask)) < mask.total());
	if (profitable && !bandView->isSeedModeEnabled()) {
		setGUIEnabled(false);
		for (size_t i = 0; i < viewers.size(); ++i) {
			tmp_sets.push_back(sets_ptr(new SharedData<std::vector<BinSet> >(NULL)));
			viewers[i]->subLabelMask(tmp_sets[i], mask);
		}
	}

	emit clearLabel();

	if (!bandView->isSeedModeEnabled()) {
		if (profitable) {
			for (size_t i = 0; i < viewers.size(); ++i) {
				viewers[i]->addLabelMask(tmp_sets[i], mask);
			}

			BackgroundTaskPtr taskEpilog(new BackgroundTask());
			QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
				this, SLOT(finishTask(bool)), Qt::QueuedConnection);
			queue.push(taskEpilog);
		} else {
			refreshLabelsInViewers();
		}
	}
}

void MainWindow::labelmask(bool negative)
{
	std::vector<sets_ptr> tmp_sets;
	cv::Mat1b mask = activeViewer->getMask();
	bool profitable = ((2 * cv::countNonZero(mask)) < mask.total());
	if (profitable) {
		setGUIEnabled(false);
		for (size_t i = 0; i < viewers.size(); ++i) {
			tmp_sets.push_back(sets_ptr(new SharedData<std::vector<BinSet> >(NULL)));
			viewers[i]->subLabelMask(tmp_sets[i], mask);
		}
	}

	emit alterLabel(activeViewer->getMask(), negative);

	if (profitable) {
		for (size_t i = 0; i < viewers.size(); ++i) {
			viewers[i]->addLabelMask(tmp_sets[i], mask);
		}

		BackgroundTaskPtr taskEpilog(new BackgroundTask());
		QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
			this, SLOT(finishTask(bool)), Qt::QueuedConnection);
		queue.push(taskEpilog);
	} else {
		refreshLabelsInViewers();
	}
}

void MainWindow::newOverlay()
{
	emit drawOverlay(activeViewer->getMask());
}

void MainWindow::reshapeDock(bool floating)
{
	SharedDataLock image_lock(image->mutex);
	if (!floating || (*image)->height == 0)
		return;

	float src_aspect = (*image)->width/(float)(*image)->height;
	float dest_aspect = bandView->width()/(float)bandView->height();
	image_lock.unlock();
	// we force the dock aspect ratio to fit band image aspect ratio.
	// this is not 100% correct
	if (src_aspect > dest_aspect) {
		bandDock->resize(bandDock->width(), bandDock->width()/src_aspect);
	} else
		bandDock->resize(bandDock->height()*src_aspect, bandDock->height());
}

QIcon MainWindow::colorIcon(const QColor &color)
{
	QPixmap pm(32, 32);
	pm.fill(color);
	return QIcon(pm);
}

bool MainWindow::haveImagePCA()
{
	return imagepca.get();
}

bool MainWindow::haveGradientPCA()
{
	return gradientpca.get();
}

void MainWindow::buildIlluminant(int temp)
{
	assert(temp > 0);
	Illuminant il(temp);
	std::vector<multi_img::Value> cf;

	SharedMultiImgBaseGuard guard(*image_lim);
	il.calcWeight((*image_lim)->meta[0].center,
				  (*image_lim)->meta[(*image_lim)->size()-1].center);
	cf = (*image_lim)->getIllumCoeff(il);
	illuminants[temp] = make_pair(il, cf);
}

const Illuminant & MainWindow::getIlluminant(int temp)
{
	assert(temp > 0);
	Illum_map::iterator i = illuminants.find(temp);
	if (i != illuminants.end())
		return i->second.first;

	buildIlluminant(temp);
	return illuminants[temp].first;
}

const std::vector<multi_img::Value> & MainWindow::getIlluminantC(int temp)
{
	assert(temp > 0);
	Illum_map::iterator i = illuminants.find(temp);
	if (i != illuminants.end())
		return i->second.second;

	buildIlluminant(temp);
	return illuminants[temp].second;
}

void MainWindow::setI1(int index) {
	int i1 = i1Box->itemData(index).value<int>();
	if (i1 > 0) {
		i1Check->setEnabled(true);
		if (i1Check->isChecked())
			viewIMG->setIlluminant(getIlluminantC(i1), false);
	} else {
		i1Check->setEnabled(false);
		std::vector<multi_img::Value> empty;
		viewIMG->setIlluminant(empty, false);
	}
}

void MainWindow::setI1Visible(bool visible)
{
	if (visible) {
		int i1 = i1Box->itemData(i1Box->currentIndex()).value<int>();
		viewIMG->setIlluminant(getIlluminantC(i1), false);
	} else {
		std::vector<multi_img::Value> empty;
		viewIMG->setIlluminant(empty, false);
	}
}

void MainWindow::loadLabeling(QString filename)
{
	QString actual_filename;
	if (!startupLabelFile.isEmpty()) {
		actual_filename = startupLabelFile;
		startupLabelFile.clear();
	} else {
		actual_filename = filename;
	}

	int height;
	int width;
	{
		SharedMultiImgBaseGuard guard(*image_lim);
		height = (*image_lim)->height;
		width = (*image_lim)->width;
	}

	IOGui io("Labeling Image File", "labeling image", this);
	cv::Mat input = io.readFile(actual_filename, -1, height, width);
	if (input.empty())
		return;

	vole::Labeling labeling(input, false);

	// if the user is operating within ROI, apply it to labeling as well */
	if (roi != cv::Rect(0, 0, width, height))
		labeling.setLabels(labeling.getLabels()(roi));

	setLabels(labeling);
}

void MainWindow::loadSeeds()
{
	int height;
	int width;
	{
		SharedMultiImgBaseGuard guard(*image_lim);
		height = (*image_lim)->height;
		width = (*image_lim)->width;
	}

	IOGui io("Seed Image File", "seed image", this);
	cv::Mat1s seeding = io.readFile(QString(), 0, height, width);
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

void MainWindow::saveLabeling()
{
	vole::Labeling labeling(bandView->labels);
	cv::Mat3b output = labeling.bgr();

	IOGui io("Labeling As Image File", "labeling image", this);
	io.writeFile(QString(), output);
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

void MainWindow::ROIDecision(QAbstractButton *sender)
{
	QDialogButtonBox::ButtonRole role = roiButtons->buttonRole(sender);
	roiButtons->setDisabled(true);

	if (role == QDialogButtonBox::ResetRole) {

		if (roi.width > 1) {
			roiView->roi = QRect(roi.x, roi.y, roi.width, roi.height);
		} else {
			// fetch image dimensions
			int height;
			int width;
			{
				SharedMultiImgBaseGuard guard(*image_lim);
				height = (*image_lim)->height;
				width = (*image_lim)->width;
			}

			// set ROI to full image
			roiView->roi = QRect(0, 0, width, height);
		}
		roiView->update();
	} else if (role == QDialogButtonBox::ApplyRole) {

		bool reuse = true;
		if (runningTask != TT_NONE) {
			queue.cancelTasks(roi);
			reuse = false;
		}

		const QRect &r = roiView->roi;
		roi = cv::Rect(r.x(), r.y(), r.width(), r.height());
		
		setGUIEnabled(false, TT_SELECT_ROI);

		applyROI(reuse);

		BackgroundTaskPtr taskEpilog(new BackgroundTask(roi));
		QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
			this, SLOT(finishTask(bool)), Qt::QueuedConnection);
		queue.push(taskEpilog);
	}
}

void MainWindow::ROISelection(const QRect &roi)
{
	roiButtons->setEnabled(true);

	QString title("<b>ROI:</b> %1, %2 - %3, %4 (%5x%6)");
	title = title.arg(roi.x()).arg(roi.y()).arg(roi.right()).arg(roi.bottom())
			.arg(roi.width()).arg(roi.height());
	roiTitle->setText(title);
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
