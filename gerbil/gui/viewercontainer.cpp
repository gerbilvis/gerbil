#include "viewercontainer.h"
//#include "ui_viewerswidget.h"

#include "viewport.h"
#include "mainwindow.h"

#include <QSignalMapper>

// TODO
//
// * check activeViewer is handled correctly


ViewerContainer::ViewerContainer(QWidget *parent)
    : QWidget(parent)
    //ui(new Ui::ViewerContainer)
{
    //ui->setupUi(this);
    initUi();
}

ViewerContainer::~ViewerContainer()
{
    //delete ui;
}

void ViewerContainer::setTaskQueue(BackgroundTaskQueue *taskQueue)
{
    this->taskQueue = taskQueue;
}

void ViewerContainer::setLabels(cv::Mat1s labels)
{
	ViewerList vl = vm.values();
	foreach(multi_img_viewer *viewer, vl) {
		viewer->labels = labels;
	}
}

void ViewerContainer::refreshLabelsInViewers()
{
	//setGUIEnabled(false);
	emit requestGUIEnabled(false, TT_NONE);
//	for (size_t i = 0; i < viewers.size(); ++i)
//		viewers[i]->updateLabels();
	ViewerList vl = vm.values();
	foreach(multi_img_viewer *viewer, vl) {
		viewer->updateLabels();
	}

	BackgroundTaskPtr taskEpilog(new BackgroundTask());
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)),
		this, SLOT(finishTask(bool)), Qt::QueuedConnection);
	taskQueue->push(taskEpilog);
}

void ViewerContainer::viewersHighlight(short)
{
	// TODO impl
}

void ViewerContainer::addImage(representation repr, sets_ptr temp,
							   const std::vector<cv::Rect> &regions,
							   cv::Rect roi)
{
	multi_img_viewer *viewer = vm.value(repr);
	if(!viewer->isPayloadHidden()) {
		viewer->addImage(temp, regions, roi);
	}
}

void ViewerContainer::subImage(representation repr, sets_ptr temp, const std::vector<cv::Rect> &regions, cv::Rect roi)
{
	multi_img_viewer *viewer = vm.value(repr);
	if(!viewer->isPayloadHidden()) {
		viewer->subImage(temp, regions, roi);
	}
}

void ViewerContainer::setImage(representation repr, SharedMultiImgPtr image, cv::Rect roi)
{
	multi_img_viewer *viewer = vm.value(repr);
	if(!viewer->isPayloadHidden()) {
		viewer->setImage(image, roi);
	}
}

void ViewerContainer::toggleLabels(bool toggle)
{
	ViewerList vl = vm.values();
	foreach(multi_img_viewer *viewer, vl) {
		viewer->toggleLabels(toggle);
	}
}

void ViewerContainer::updateLabelColors(QVector<QColor> labelColors, bool changed)
{
	ViewerList vl = vm.values();
	foreach(multi_img_viewer *viewer, vl) {
		viewer->updateLabelColors(labelColors, changed);
	}
}

void ViewerContainer::updateBinning(representation repr, int bins)
{
	multi_img_viewer *viewer = vm.value(repr);
	viewer->updateBinning(bins);
}

void ViewerContainer::updateViewerBandSelections(int numbands)
{
	ViewerList vl = vm.values();
	foreach(multi_img_viewer *viewer, vl) {
		if (viewer->getSelection() >= numbands)
			viewer->setSelection(0);
	}
}

size_t ViewerContainer::size() const
{
	return vm.size();
}

const QPixmap *ViewerContainer::getBand(representation repr, int dim)
{
	std::vector<QPixmap*> &v = (*bands)[repr];

	if (!v[dim]) {
		multi_img_viewer *viewer = vm.value(repr);
		SharedMultiImgPtr multi = viewer->getImage();
		qimage_ptr qimg(new SharedData<QImage>(new QImage()));

		SharedDataLock hlock(multi->mutex);

		BackgroundTaskPtr taskConvert(new MultiImg::Band2QImageTbb(multi, qimg, dim));
		taskConvert->run();

		hlock.unlock();

		v[dim] = new QPixmap(QPixmap::fromImage(**qimg));
	}
	return v[dim];
}

int ViewerContainer::getSelection(representation repr)
{
	multi_img_viewer *viewer = vm.value(repr);
	return viewer->getSelection();
}

SharedMultiImgPtr ViewerContainer::getViewerImage(representation repr)
{
	multi_img_viewer *viewer = vm.value(repr);
	return viewer->getImage();
}

representation ViewerContainer::getActiveRepresentation() const
{
	// TODO impl
}

void ViewerContainer::setIlluminant(representation repr,
									const std::vector<multi_img_base::Value> &illuminant,
									bool for_real)
{
	multi_img_viewer *viewer = vm.value(repr);
	viewer->setIlluminant(illuminant, for_real);
}

void ViewerContainer::setGUIEnabled(bool enable, TaskType tt)
{
	ViewerList vl = vm.values();
    foreach(multi_img_viewer *viewer, vl) {
        viewer->enableBinSlider(enable);
        viewer->setEnabled(enable || tt == TT_BIN_COUNT || tt == TT_TOGGLE_VIEWER);
    }
}

void ViewerContainer::toggleViewer(bool enable, representation repr)
{
	if(enable)
		toggleViewerEnable(repr);
	else
		toggleViewerDisable(repr);
}

void ViewerContainer::newROI(cv::Rect roi)
{
	this->roi = roi;
}

void ViewerContainer::setActiveViewer(int repr)
{
	representation r = static_cast<representation>(repr);
	if (vm.value(r)->getImage().get()) {
		activeViewer = vm.value(r);
	} else {
		activeViewer = vm.value(IMG);
	}
}

void ViewerContainer::imgCalculationComplete(bool success)
{
	if (success)
		finishViewerRefresh(IMG);
}

void ViewerContainer::gradCalculationComplete(bool success)
{
	if (success)
		finishViewerRefresh(GRAD);
}

void ViewerContainer::imgPcaCalculationComplete(bool success)
{
	if (success)
		finishViewerRefresh(IMGPCA);
}

void ViewerContainer::gradPcaCalculationComplete(bool success)
{
	if (success)
		finishViewerRefresh(GRADPCA);
}

void ViewerContainer::finishViewerRefresh(representation repr)
{
	multi_img_viewer *viewer = vm.value(repr);
	viewer->setEnabled(true);
	connect(this, SIGNAL(viewersOverlay(int,int)),
		viewer, SLOT(overlay(int, int)));
	connect(this, SIGNAL(viewportsKillHover()),
		viewer->getViewport(), SLOT(killHover()));
	connect(this, SIGNAL(viewersSubPixels(std::map<std::pair<int,int>,short>)),
		viewer, SLOT(subPixels(const std::map<std::pair<int, int>, short> &)));
	connect(this, SIGNAL(viewersAddPixels(std::map<std::pair<int,int>,short>)),
		viewer, SLOT(addPixels(const std::map<std::pair<int, int>, short> &)));
	if (repr == GRAD) {
		emit normTargetChanged(true);
	}
	if (activeViewer->getType() == repr) {
		emit bandUpdateNeeded(activeViewer->getType(),
							  activeViewer->getSelection());
	}
}

void ViewerContainer::disconnectViewer(representation repr)
{
	multi_img_viewer *viewer = vm.value(repr);
	disconnect(this, SIGNAL(viewersOverlay(int,int)),
		viewer, SLOT(overlay(int, int)));
	disconnect(this, SIGNAL(viewportsKillHover()),
		viewer->getViewport(), SLOT(killHover()));
	disconnect(this, SIGNAL(viewersSubPixels(std::map<std::pair<int,int>,short>)),
		viewer, SLOT(subPixels(const std::map<std::pair<int, int>, short> &)));
	disconnect(this, SIGNAL(viewersAddPixels(std::map<std::pair<int,int>,short>)),
		viewer, SLOT(addPixels(const std::map<std::pair<int, int>, short> &)));
}

void ViewerContainer::disconnectAllViewers()
{
	ViewerList vl = vm.values();
	foreach(multi_img_viewer *viewer, vl) {
		disconnectViewer(viewer->getType());
	}
}

void ViewerContainer::finishTask(bool success)
{
	if(success)
		emit requestGUIEnabled(true, TT_NONE);
}

void ViewerContainer::finishNormRangeImgChange(bool success)
{
	if (success) {
		SharedDataLock hlock((*image)->mutex);
		(*bands)[GRAD].assign((**image)->size(), NULL);
		hlock.unlock();
		emit bandUpdateNeeded(
					IMG,
					vm.value(IMG)->getSelection()
					);
	}
}


void ViewerContainer::finishNormRangeGradChange(bool success)
{
	if (success) {
		SharedDataLock hlock((*gradient)->mutex);
		(*bands)[GRAD].assign((**gradient)->size(), NULL);
		hlock.unlock();
		emit bandUpdateNeeded(
					GRAD,
					vm.value(GRAD)->getSelection()
					);
	}
}

void ViewerContainer::toggleViewerEnable(representation repr)
{
	disconnectViewer(repr);

	switch(repr) {
	case IMG:
		break;
	case GRAD:
		break;
	case IMGPCA:
	{
		multi_img_viewer *viewer = vm.value(IMGPCA);
		viewer->resetImage();
		emit imageResetNeeded(IMGPCA);
		if(activeViewer == viewer) {
			viewer->activateViewport();
			emit bandUpdateNeeded(viewer->getType(),
								  viewer->getSelection());
		}
	}
		break;
	case GRADPCA:
	{
		multi_img_viewer *viewer = vm.value(GRADPCA);
		viewer->resetImage();
		emit imageResetNeeded(GRADPCA);
		if(activeViewer == viewer) {
			viewer->activateViewport();
			emit bandUpdateNeeded(viewer->getType(),
								  viewer->getSelection());
		}
	}
		break;
	default:
		assert(false);
		break;
	}
}

void ViewerContainer::toggleViewerDisable(representation repr)
{
	multi_img_viewer *viewer = vm.value(repr);

	emit requestGUIEnabled(false, TT_TOGGLE_VIEWER);

	switch(repr) {
	case IMG:
	{
		viewer->setImage(*image, roi);
		BackgroundTaskPtr task(new BackgroundTask(roi));
		QObject::connect(task.get(), SIGNAL(finished(bool)),
			this, SLOT(imgCalculationComplete(bool)), Qt::QueuedConnection);
		taskQueue->push(task);
	}
		break;
	case GRAD:
	{
		viewer->setImage(*gradient, roi);
		BackgroundTaskPtr task(new BackgroundTask(roi));
		QObject::connect(task.get(), SIGNAL(finished(bool)),
			this, SLOT(gradCalculationComplete(bool)), Qt::QueuedConnection);
		taskQueue->push(task);
	}
		break;
	case IMGPCA:
	{
		// FIXME very bad style to access member of MainWindow
		imagepca->reset(new SharedMultiImgBase(new multi_img(0, 0, 0)));

		BackgroundTaskPtr taskPca(new MultiImg::PcaTbb(
			*image, *imagepca, 0, roi));
		taskQueue->push(taskPca);

		viewer->setImage(*imagepca, roi);

		BackgroundTaskPtr task(new BackgroundTask(roi));
		QObject::connect(task.get(), SIGNAL(finished(bool)),
			this, SLOT(imgPcaCalculationComplete(bool)), Qt::QueuedConnection);
		taskQueue->push(task);

	}
		break;
	case GRADPCA:
	{
		// FIXME very bad style to access member of MainWindow
		gradientpca->reset(new SharedMultiImgBase(new multi_img(0, 0, 0)));

		BackgroundTaskPtr taskPca(new MultiImg::PcaTbb(
			*gradient, *gradientpca, 0, roi));
		taskQueue->push(taskPca);

		viewer->setImage(*imagepca, roi);

		BackgroundTaskPtr task(new BackgroundTask(roi));
		QObject::connect(task.get(), SIGNAL(finished(bool)),
			this, SLOT(gradPcaCalculationComplete(bool)), Qt::QueuedConnection);
		taskQueue->push(task);

	}
		break;
	default:
		assert(false);
		break;
	} // switch


	BackgroundTaskPtr taskEpilog(new BackgroundTask(roi));
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)),
		this, SLOT(finishTask(bool)), Qt::QueuedConnection);
	taskQueue->push(taskEpilog);
}


//void ViewerContainer::clearBinSets(const std::vector<cv::Rect>& sub, const cv::Rect& roi )
//{
//	ViewerList viewers;


//	sets_ptr tmp_sets_imagepca(new SharedData<std::vector<BinSet> >(NULL));
//	sets_ptr tmp_sets_gradient(new SharedData<std::vector<BinSet> >(NULL));
//	sets_ptr tmp_sets_gradientpca(new SharedData<std::vector<BinSet> >(NULL));

////    for(int repr=IMG; repr<REPSIZE; repr++) {
////        viewers = vm.values(repr);
////        foreach (multi_img_viewer *v, viewers) {
////            sets_ptr tmp_sets(new SharedData<std::vector<BinSet> >(NULL));
////            if(!v->isPayloadHidden())
////                v->subImage(tmp_sets, sub, roi);
////        }
////    }


//}

void ViewerContainer::initUi()
{
    vLayout = new QVBoxLayout(this);

	// CAVEAT: Only one viewer per representation type is supported.
    createViewer(IMG);
    createViewer(GRAD);
    createViewer(IMGPCA);
    createViewer(GRADPCA);

	// start with IMG, hide IMGPCA, GRADPCA at the beginning
	activeViewer = vm.value(IMG);
	vm.value(IMG)->setActive();
	vm.value(IMGPCA)->toggleFold();
	vm.value(GRAD)->toggleFold();


	ViewerList vl = vm.values();
	// for self-activation of viewports
	QSignalMapper *vpsmap = new QSignalMapper(this);
	for (size_t i = 0; i < vl.size(); ++i) {
		Viewport *viewport = vl[i]->getViewport();
		vpsmap->setMapping(viewport, (int)i);
		connect(viewport, SIGNAL(activated()),
				vpsmap, SLOT(map()));
	}
	connect(vpsmap, SIGNAL(mapped(int)),
			this, SLOT(setActiveViewer(int)));

	for (size_t i = 0; i < vl.size(); ++i) {
		multi_img_viewer *viewer1 = vl[i];
		Viewport *viewport1 = viewer1->getViewport();

		// connect pass through signals from BandView
		// TODO these were originally conditionally: if (!v->isPayloadHidden()) {},
		//      need to push hidden state handling into multi_img_viewer.
		connect(this, SIGNAL(viewportsKillHover()),
				viewport1, SLOT(killHover()));
		connect(this, SIGNAL(viewersOverlay(int,int)),
				viewer1, SLOT(overlay(int,int)));
		connect(this, SIGNAL(viewersSubPixels(std::map<std::pair<int,int>,short>)),
				viewer1, SLOT(subPixels(std::map<std::pair<int,int>,short>)));
		connect(this, SIGNAL(viewersAddPixels(std::map<std::pair<int,int>,short>)),
				viewer1, SLOT(addPixels(std::map<std::pair<int,int>,short>)));

		// connect pass through signals from markButton
		connect(this, SIGNAL(viewersToggleLabeled(bool)),
				viewer1, SLOT(toggleLabeled(bool)));
		// connect pass through signals from nonmarkButton
		connect(this, SIGNAL(viewersToggleUnlabeled(bool)),
				viewer1, SLOT(toggleUnlabeled(bool)));

		connect(viewport1, SIGNAL(bandSelected(representation, int)),
				this, SIGNAL(viewportBandSelected(representation,int)));

		connect(viewer1, SIGNAL(setGUIEnabled(bool, TaskType)),
				this, SIGNAL(viewerSetGUIEnabled(bool, TaskType)));
		connect(viewer1, SIGNAL(finishTask(bool)),
				this, SIGNAL(viewerFinishTask(bool)));

		connect(viewer1, SIGNAL(newOverlay()),
				this, SLOT(newOverlay()));
		connect(viewport1, SIGNAL(newOverlay(int)),
				this, SIGNAL(viewportNewOverlay(int)));

		connect(viewport1, SIGNAL(addSelection()),
				this, SIGNAL(viewportAddToLabel()));
		connect(viewport1, SIGNAL(remSelection()),
				this, SIGNAL(viewportRemFromLabel()));

		// non-pass-through
		connect(viewer1, SIGNAL(toggleViewer(bool, representation)),
				this, SLOT(toggleViewer(bool, representation)));

		for (size_t j = 0; j < vl.size(); ++j) {
			multi_img_viewer *viewer2 = vl[j];
			const Viewport *viewport2 = viewer2->getViewport();
			// connect folding signal to all viewports
			connect(viewer1, SIGNAL(folding()),
					viewport2, SLOT(folding()));

			// connect activation signal to all *other* viewers
			if (i != j) {
				connect(viewport1, SIGNAL(activated()),
						viewer2, SLOT(setInactive()));
			}
		}
	}
}

multi_img_viewer *ViewerContainer::createViewer(representation repr)
{
    multi_img_viewer *viewer = new multi_img_viewer(this);
    viewer->setType(repr);
    viewer->queue = taskQueue;
    vm.insert(repr, viewer);
    vLayout->addWidget(viewer);
}

void ViewerContainer::newOverlay()
{
	emit drawOverlay(activeViewer->getMask());
}

//void ViewerContainer::addViewersLabelMask(sets_ptr temp, const cv::Mat1b &mask)
//{
//	ViewerList vl = vm.values();
//	foreach(multi_img_viewer *viewer, vl) {
//		viewer->addLabelMask(temp, mask);
//	}
//}

//void ViewerContainer::subViewersLabelMask(sets_ptr temp, const cv::Mat1b &mask)
//{
//	ViewerList vl = vm.values();
//	foreach(multi_img_viewer *viewer, vl) {
//		viewer->subLabelMask(temp, mask);
//	}
//}


void ViewerContainer::labelflush(bool seedModeEnabled, short curLabel)
{
	std::vector<sets_ptr> tmp_sets;
	ViewerList vl = vm.values();
	cv::Mat1b mask(labels->rows, labels->cols);
	mask = (*labels == curLabel);
	bool profitable = ((2 * cv::countNonZero(mask)) < mask.total());
	if (profitable && !seedModeEnabled) {
		// setGUIEnabled(false);
		emit requestGUIEnabled(false, TT_NONE);
		for (size_t i = 0; i < vl.size(); ++i) {
			tmp_sets.push_back(sets_ptr(new SharedData<std::vector<BinSet> >(NULL)));
			vl[i]->subLabelMask(tmp_sets[i], mask);
		}
	}

	emit clearLabel();

	if (seedModeEnabled) {
		if (profitable) {
			for (size_t i = 0; i < vl.size(); ++i) {
				vl[i]->addLabelMask(tmp_sets[i], mask);
			}

			BackgroundTaskPtr taskEpilog(new BackgroundTask());
			QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)),
				this, SLOT(finishTask(bool)), Qt::QueuedConnection);
			taskQueue->push(taskEpilog);
		} else {
			refreshLabelsInViewers();
		}
	}
}

void ViewerContainer::labelmask(bool negative)
{
	std::vector<sets_ptr> tmp_sets;
	ViewerList vl = vm.values();
	cv::Mat1b mask = activeViewer->getMask();
	bool profitable = ((2 * cv::countNonZero(mask)) < mask.total());
	if (profitable) {
//		setGUIEnabled(false);
		emit requestGUIEnabled(false, TT_NONE);
		for (size_t i = 0; i < vl.size(); ++i) {
			tmp_sets.push_back(sets_ptr(new SharedData<std::vector<BinSet> >(NULL)));
			vl[i]->subLabelMask(tmp_sets[i], mask);
		}
	}

	// TODO propagate
	emit alterLabel(activeViewer->getMask(), negative);

	if (profitable) {
		for (size_t i = 0; i < vl.size(); ++i) {
			vl[i]->addLabelMask(tmp_sets[i], mask);
		}

		BackgroundTaskPtr taskEpilog(new BackgroundTask());
		QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)),
			this, SLOT(finishTask(bool)), Qt::QueuedConnection);
		taskQueue->push(taskEpilog);
	} else {
		refreshLabelsInViewers();
	}
}
