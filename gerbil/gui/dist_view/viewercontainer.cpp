#include "viewercontainer.h"

#include "viewport.h"
#include "../mainwindow.h"

#include <QSignalMapper>

// for debug msgs
#include <boost/format.hpp>
#include <QDebug>

// TODO
//
// * check activeViewer is handled correctly

#define GGDBG_REPR(repr) GGDBGM(format("%1%")%repr << endl)

ViewerContainer::ViewerContainer(QWidget *parent)
	: QWidget(parent)
{
}

ViewerContainer::~ViewerContainer()
{
	// nothing
}

void ViewerContainer::setTaskQueue(BackgroundTaskQueue *taskQueue)
{
    this->taskQueue = taskQueue;
}

void ViewerContainer::setLabelMatrix(cv::Mat1s matrix)
{
	ViewerList vl = vm.values();
	foreach(multi_img_viewer *viewer, vl) {
		viewer->labels = matrix;
	}
}

void ViewerContainer::updateLabels()
{
	//setGUIEnabled(false);
	emit requestGUIEnabled(false, TT_NONE);
	ViewerList vl = vm.values();
	foreach(multi_img_viewer *viewer, vl) {
		viewer->updateLabels();
	}

	// re-enable gui
	BackgroundTaskPtr taskEpilog(new BackgroundTask());
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)),
		this, SLOT(finishTask(bool)), Qt::QueuedConnection);
	taskQueue->push(taskEpilog);
}

void ViewerContainer::newIlluminant(cv::Mat1f illum)
{
	multi_img_viewer *viewer = vm.value(IMG);
	viewer->setIlluminant(illum);
}

void ViewerContainer::showIlluminationCurve(bool show)
{
	//GGDBGM(show <<endl);
	multi_img_viewer *viewer = vm.value(IMG);
	viewer->showIlluminationCurve(show);
}

void ViewerContainer::setIlluminantApplied(bool applied)
{
	multi_img_viewer *viewer = vm.value(IMG);
	viewer->setIlluminantIsApplied(applied);
}

void ViewerContainer::addImage(representation repr, sets_ptr temp,
							   const std::vector<cv::Rect> &regions,
							   cv::Rect roi)
{
	multi_img_viewer *viewer = vm.value(repr);
	assert(viewer);
	if(!viewer->isPayloadHidden()) {
		GGDBG_REPR(repr);
		viewer->addImage(temp, regions, roi);
	}
}

void ViewerContainer::subImage(representation repr, sets_ptr temp, const std::vector<cv::Rect> &regions, cv::Rect roi)
{
	multi_img_viewer *viewer = vm.value(repr);
	assert(viewer);
	if(!viewer->isPayloadHidden()) {
		GGDBG_REPR(repr);
		viewer->subImage(temp, regions, roi);
	}
}

void ViewerContainer::setImage(representation repr, SharedMultiImgPtr image, cv::Rect roi)
{
	multi_img_viewer *viewer = vm.value(repr);
	assert(viewer);
	//GGDBGM(format("repr=%1%, image.get()=%2%)\n")
	//	   % repr % image.get());
	if(!viewer->isPayloadHidden()) {
		GGDBG_REPR(repr);
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

void ViewerContainer::updateLabelColors(QVector<QColor> colors, bool changed)
{
	ViewerList vl = vm.values();
	foreach(multi_img_viewer *viewer, vl) {
		viewer->updateLabelColors(colors, changed);
	}
}

void ViewerContainer::updateBinning(representation repr, int bins)
{
	GGDBG_REPR(repr);
	multi_img_viewer *viewer = vm.value(repr);
	viewer->updateBinning(bins);
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
	assert(activeViewer);
	return vm.key(activeViewer);
}

const cv::Mat1b ViewerContainer::getHighlightMask() const
{
	return activeViewer->getHighlightMask();
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
	//GGDBGM(format("enable=%1%, tt=%2%\n") % enable % tt);
	ViewerList vl = vm.values();
	foreach(multi_img_viewer *viewer, vl) {
        viewer->setEnabled(enable || tt == TT_BIN_COUNT || tt == TT_TOGGLE_VIEWER);
		//GGDBGM(format("enable=%1% viewer=%2% %3%\n")
		//			 %enable %viewer %viewer->getType());
    }

}

void ViewerContainer::toggleViewer(bool enable, representation repr)
{
	GGDBGM(format("toggle=%1% representation=%2%\n") %enable % repr);
	if(enable)
		enableViewer(repr);
	else
		disableViewer(repr);
}

void ViewerContainer::newROI(cv::Rect roi)
{
	this->roi = roi;
}

void ViewerContainer::setActiveViewer(int repr)
{
	representation r = static_cast<representation>(repr);
	GGDBG_REPR(repr);
	if (vm.value(r)->getImage().get()) {
		activeViewer = vm.value(r);
	} else {
		activeViewer = vm.value(IMG);
	}
}

void ViewerContainer::imgCalculationComplete(bool success)
{
	if (success && !vm.value(IMG)->isPayloadHidden())
		finishViewerRefresh(IMG);
}

void ViewerContainer::gradCalculationComplete(bool success)
{
	if (success && !vm.value(GRAD)->isPayloadHidden())
		finishViewerRefresh(GRAD);
}

void ViewerContainer::imgPcaCalculationComplete(bool success)
{
	if (success && !vm.value(IMGPCA)->isPayloadHidden())
		finishViewerRefresh(IMGPCA);
}

void ViewerContainer::gradPcaCalculationComplete(bool success)
{
	if (success && !vm.value(GRADPCA)->isPayloadHidden())
		finishViewerRefresh(GRADPCA);
}

void ViewerContainer::finishViewerRefresh(representation repr)
{
	GGDBGM(format("representation %1%\n") % repr);
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
		emit bandSelected(activeViewer->getType(),
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
	/* TODO: this is *so* wrong. what this does is after changing the range
	 * ensure that band cache is deleted and new band is shown
	 * however bandSelected will select this representation, even if it was not
	 * selected before. also this should be done by the image model
	 * and the image model has other means of doing this (empty the map)
	 */
	if (success) {
		SharedDataLock hlock((*image)->mutex);
		(*bands)[IMG].assign((**image)->size(), NULL);
		hlock.unlock();
		emit bandSelected(IMG, vm.value(IMG)->getSelection());
	}
}


void ViewerContainer::finishNormRangeGradChange(bool success)
{
	// ************** see above
	if (success) {
		SharedDataLock hlock((*gradient)->mutex);
		(*bands)[GRAD].assign((**gradient)->size(), NULL);
		hlock.unlock();
		emit bandSelected(GRAD,	vm.value(GRAD)->getSelection());
	}
}

void ViewerContainer::disableViewer(representation repr)
{
	// TODO: where is it connected back?
	disconnectViewer(repr);

	/* TODO: why do we have to differentiate between IMG/GRAD and PCA variants?
	 * probably img and grad are keep-alive, the others not. in the future
	 * we do not want this differentiation, as other parties will ALSO
	 * access the PCA reps. this will be part of ImageModel housekeeping! */
	switch(repr) {
	case IMG:
	case GRAD:
		break;

	case IMGPCA:
	case GRADPCA:
	{
		multi_img_viewer *viewer = vm.value(repr);
		// TODO: here we would kill the image data. TODO: Do that in ImageModel!
		// viewer->resetImage();

		// TODO: what does this do?! I mean, WHY?
		if(activeViewer == viewer) {
			viewer->activateViewport();
			emit bandSelected(viewer->getType(), viewer->getSelection());
		}
	}
		break;
	default:
		assert(false);
		break;
	}

	bool allFolded = true;
	foreach(multi_img_viewer *v, vm.values()) {
		if(!v->isPayloadHidden()) {
			allFolded = false;
			break;
		}
	}
}

void ViewerContainer::enableViewer(representation repr)
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

		viewer->setImage(*gradientpca, roi);

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

void ViewerContainer::initUi()
{
	// Only one viewer per representation type is supported.
    createViewer(IMG);
	createViewer(GRAD);
    createViewer(IMGPCA);
    createViewer(GRADPCA);

	// start with IMG, hide IMGPCA, GRADPCA at the beginning
	activeViewer = vm.value(IMG);
	activeViewer->setActive();
	vm.value(IMGPCA)->toggleFold();
	vm.value(GRADPCA)->toggleFold();

	// create layout and fill with viewers
	QVBoxLayout *vLayout = new QVBoxLayout(this);
	ViewerList vl = vm.values();
	foreach(multi_img_viewer *viewer, vl) {
		// add with stretch = 1 so they will stay evenly distributed in space
		vLayout->addWidget(viewer, 1);
	}
	vLayout->addStretch(); // align on top when all folded

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
		//GGDBGM(boost::format("viewer %1% isPayloadHidden()=%2%\n")
		//					%i %viewer1->isPayloadHidden());
		if(!viewer1->isPayloadHidden()) {
			connect(this, SIGNAL(viewportsKillHover()),
					viewport1, SLOT(killHover()));
			connect(this, SIGNAL(viewersOverlay(int,int)),
					viewer1, SLOT(overlay(int,int)));
			connect(this, SIGNAL(viewersSubPixels(std::map<std::pair<int,int>,short>)),
					viewer1, SLOT(subPixels(std::map<std::pair<int,int>,short>)));
			connect(this, SIGNAL(viewersAddPixels(std::map<std::pair<int,int>,short>)),
					viewer1, SLOT(addPixels(std::map<std::pair<int,int>,short>)));
		}

		// connect pass through signals from markButton
		connect(this, SIGNAL(viewersToggleLabeled(bool)),
				viewer1, SLOT(toggleLabeled(bool)));
		// connect pass through signals from nonmarkButton
		connect(this, SIGNAL(viewersToggleUnlabeled(bool)),
				viewer1, SLOT(toggleUnlabeled(bool)));

		// todo make this globally consistent
		connect(viewer1, SIGNAL(finishTask(bool)),
				this, SLOT(finishTask(bool)));

		/* following stuff is used to pass to/from MainWindow / Controller */

		connect(viewport1, SIGNAL(bandSelected(representation, int)),
				this, SIGNAL(bandSelected(representation, int)));

		connect(viewer1, SIGNAL(setGUIEnabled(bool, TaskType)),
				this, SIGNAL(requestGUIEnabled(bool,TaskType)));

		connect(viewport1, SIGNAL(addSelection()),
				this, SIGNAL(viewportAddSelection()));
		connect(viewport1, SIGNAL(remSelection()),
				this, SIGNAL(viewportRemSelection()));
		connect(this, SIGNAL(viewersHighlight(short)),
				viewport1, SLOT(highlight(short)));

		// overlay for bandview (we emit drawOverlay)
		connect(viewer1, SIGNAL(newOverlay()),
				this, SLOT(newOverlay()));
		connect(viewport1, SIGNAL(newOverlay(int)),
				this, SLOT(newOverlay()));

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
	viewer->setSizePolicy(QSizePolicy::Preferred, // hor
						  QSizePolicy::Expanding); // ver
    viewer->setType(repr);
	assert(taskQueue);
    viewer->queue = taskQueue;
    vm.insert(repr, viewer);
}

void ViewerContainer::newOverlay()
{
	emit drawOverlay(activeViewer->getHighlightMask());
}

void ViewerContainer::updateLabelsPartially(cv::Mat1b mask, cv::Mat1s old)
{
	// is it worth it to do it incrementally (2 updates for each positive entry)
	bool profitable = ((2 * cv::countNonZero(mask)) < mask.total());
	if (profitable) {
		// gui disable
		emit requestGUIEnabled(false, TT_NONE);

		ViewerList vl = vm.values();
		for (size_t i = 0; i < vl.size(); ++i) {
			vl[i]->updateLabelsPartially(mask, old);
		}

		// gui enable when done
		BackgroundTaskPtr taskEpilog(new BackgroundTask());
		QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)),
			this, SLOT(finishTask(bool)), Qt::QueuedConnection);
		taskQueue->push(taskEpilog);
	} else {
		// just update the whole thing
		updateLabels();
	}
}
