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

#define GGDBG_REPR(type) GGDBGM(format("%1%")%type << endl)

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
	foreach (multi_img_viewer *viewer, map) {
		viewer->queue = taskQueue;
	}
}

void ViewerContainer::newIlluminant(cv::Mat1f illum)
{
	multi_img_viewer *viewer = map.value(representation::IMG);
	viewer->setIlluminant(illum);
}

void ViewerContainer::showIlluminationCurve(bool show)
{
	//GGDBGM(show <<endl);
	multi_img_viewer *viewer = map.value(representation::IMG);
	viewer->showIlluminationCurve(show);
}

void ViewerContainer::setIlluminantApplied(bool applied)
{
	multi_img_viewer *viewer = map.value(representation::IMG);
	viewer->setIlluminantIsApplied(applied);
}

sets_ptr ViewerContainer::subImage(representation::t type,
							   const std::vector<cv::Rect> &regions,
							   cv::Rect newRoi)
{
	// the roi change is effectively global
	roi = newRoi;

	sets_ptr temp(new SharedData<std::vector<BinSet> >(NULL));
	multi_img_viewer *viewer = map.value(type);
	assert(viewer);
	if(!viewer->isPayloadHidden()) {
		GGDBG_REPR(type);
		viewer->subImage(temp, regions, roi);
	}
	return temp;
}

void ViewerContainer::addImage(representation::t type, sets_ptr temp,
							   const std::vector<cv::Rect> &regions,
							   cv::Rect newRoi)
{
	// the roi change is effectively global
	roi = newRoi;

	multi_img_viewer *viewer = map.value(type);
	assert(viewer);
	if(!viewer->isPayloadHidden()) {
		//GGDBG_REPR(type);
		viewer->addImage(temp, regions, roi);
	}
}

void ViewerContainer::setImage(representation::t type, SharedMultiImgPtr image,
							   cv::Rect newRoi)
{
	// the roi change is effectively global
	roi = newRoi;

	multi_img_viewer *viewer = map.value(type);
	assert(viewer);
	//GGDBGM(format("type=%1%, image.get()=%2%)\n")
	//	   % type % image.get());
	if(!viewer->isPayloadHidden()) {
		//GGDBG_REPR(type);
		viewer->setImage(image, roi);
	}
}

// todo: just connect signals
void ViewerContainer::toggleLabels(bool toggle)
{
	foreach (multi_img_viewer *viewer, map) {
		viewer->toggleLabels(toggle);
	}
}

// TODO: connect-through signals
void ViewerContainer::updateLabels(const cv::Mat1s& labels,
								   const QVector<QColor> &colors,
								   bool colorsChanged)
{
	foreach(multi_img_viewer *viewer, map) {
		viewer->updateLabels(labels, colors, colorsChanged);
	}
}

void ViewerContainer::updateBinning(representation::t type, int bins)
{
	GGDBG_REPR(type);
	multi_img_viewer *viewer = map.value(type);
	viewer->updateBinning(bins);
}

int ViewerContainer::getSelection(representation::t type)
{
	multi_img_viewer *viewer = map.value(type);
	return viewer->getSelection();
}

SharedMultiImgPtr ViewerContainer::getViewerImage(representation::t type)
{
	multi_img_viewer *viewer = map.value(type);
	return viewer->getImage();
}

representation::t ViewerContainer::getActiveRepresentation() const
{
	assert(activeViewer);
	return map.key(activeViewer);
}

const cv::Mat1b ViewerContainer::getHighlightMask() const
{
	return activeViewer->getHighlightMask();
}

void ViewerContainer::setGUIEnabled(bool enable, TaskType tt)
{
	//GGDBGM(format("enable=%1%, tt=%2%\n") % enable % tt);
	foreach(multi_img_viewer *viewer, map) {
        viewer->setEnabled(enable || tt == TT_BIN_COUNT || tt == TT_TOGGLE_VIEWER);
		//GGDBGM(format("enable=%1% viewer=%2% %3%\n")
		//			 %enable %viewer %viewer->getType());
    }

}

void ViewerContainer::toggleViewer(bool enable, representation::t type)
{
	//GGDBGM(format("toggle=%1% representation=%2%\n") %enable % type);
	if(enable)
		enableViewer(type);
	else
		disableViewer(type);
}

void ViewerContainer::newROI(cv::Rect roi)
{
	this->roi = roi;
}

void ViewerContainer::setActiveViewer(int type)
{
	representation::t r = static_cast<representation::t>(type);
	//GGDBG_REPR(type);
	if (map.value(r)->getImage().get()) {
		activeViewer = map.value(r);
	} else {
		activeViewer = map.value(representation::IMG);
	}
}

void ViewerContainer::connectViewer(representation::t type)
{
	//GGDBGM(format("representation %1%\n") % type);
	multi_img_viewer *viewer = map.value(type);
	/* TODO: (code was in multi_img_viewer class */
	/*{
		SharedDataLock imglock(image->mutex);
		if (viewport->selection > (*image)->size())
			viewport->selection = 0;
	}*/

	if (viewer->isPayloadHidden())
		return;

	viewer->setEnabled(true);
	connect(this, SIGNAL(viewersOverlay(int,int)),
		viewer, SLOT(overlay(int, int)));
	connect(this, SIGNAL(viewportsKillHover()),
		viewer->getViewport(), SLOT(killHover()));
	connect(this, SIGNAL(viewersSubPixels(std::map<std::pair<int,int>,short>)),
		viewer, SLOT(subPixels(const std::map<std::pair<int, int>, short> &)));
	connect(this, SIGNAL(viewersAddPixels(std::map<std::pair<int,int>,short>)),
		viewer, SLOT(addPixels(const std::map<std::pair<int, int>, short> &)));

	// re-announce currently selected band to ensure update.
	if (activeViewer->getType() == type) {
		emit bandSelected(activeViewer->getType(),
						  activeViewer->getSelection());
	}
}

void ViewerContainer::disconnectViewer(representation::t type)
{
	multi_img_viewer *viewer = map.value(type);
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
	foreach(multi_img_viewer *viewer, map) {
		disconnectViewer(viewer->getType());
	}
}

// TODO: see how Controller does this.
void ViewerContainer::finishTask(bool success)
{
	if(success)
		emit setGUIEnabledRequested(true, TT_NONE);
}

void ViewerContainer::finishNormRangeImgChange(bool success)
{
	/* TODO: this is *so* wrong. what this does is after changing the range
	 * ensure that band cache is deleted and new band is shown
	 * however bandSelected will select this representation, even if it was not
	 * selected before. also this should be done by the image model
	 * and the image model has other means of doing this (empty the map)
	 */
/*	if (success) {
		SharedDataLock hlock((*image)->mutex);
		(*bands)[representation::IMG].assign((**image)->size(), NULL);
		hlock.unlock();
		emit bandSelected(representation::IMG, map.value(representation::IMG)->getSelection());
	}*/
}


void ViewerContainer::finishNormRangeGradChange(bool success)
{
	// ************** see above
/*	if (success) {
		SharedDataLock hlock((*gradient)->mutex);
		(*bands)[GRAD].assign((**gradient)->size(), NULL);
		hlock.unlock();
		emit bandSelected(GRAD,	map.value(GRAD)->getSelection());
	}*/
}

void ViewerContainer::disableViewer(representation::t type)
{
	// TODO: where is it connected back? (probably at reconnectViewer)
	disconnectViewer(type);

	/* TODO: why do we have to differentiate between IMG/GRAD and PCA variants?
	 * probably img and grad are keep-alive, the others not. in the future
	 * we do not want this differentiation, as other parties will ALSO
	 * access the PCA reps. this will be part of ImageModel housekeeping! */
	switch(type) {
	case representation::IMG:
	case representation::GRAD:
		break;

	case representation::IMGPCA:
	case representation::GRADPCA:
	{
		multi_img_viewer *viewer = map.value(type);
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
	foreach (multi_img_viewer *v, map.values()) {
		if(!v->isPayloadHidden()) {
			allFolded = false;
			break;
		}
	}
}

// TODO: this whole function is a joke, right?!
// the proper way is to signal that data is needed. the model should then
// deliver it and reconnectViewer() will be called or sth.
void ViewerContainer::enableViewer(representation::t type)
{
/*	multi_img_viewer *viewer = map.value(type);

	emit requestGUIEnabled(false, TT_TOGGLE_VIEWER);

	switch(type) {
	case representation::IMG:
	{
		viewer->setImage(image, roi);
		BackgroundTaskPtr task(new BackgroundTask(roi));
		QObject::connect(task.get(), SIGNAL(finished(bool)),
			this, SLOT(imgCalculationComplete(bool)), Qt::QueuedConnection);
		taskQueue->push(task);
	}
	break;
	case GRAD:
	{
		viewer->setImage(gradient, roi);
		BackgroundTaskPtr task(new BackgroundTask(roi));
		QObject::connect(task.get(), SIGNAL(finished(bool)),
			this, SLOT(gradCalculationComplete(bool)), Qt::QueuedConnection);
		taskQueue->push(task);
	}
		break;
	case IMGPCA:
	{
		// FIXME this only invalidates local pointer, EMIT A SIGNAL INSTEAD!
		// (SORRY OTHER THING WAS *REALLY* TOO NASTY)
		imagepca.reset(new SharedMultiImgBase(new multi_img(0, 0, 0)));

		BackgroundTaskPtr taskPca(new MultiImg::PcaTbb(
			image, imagepca, 0, roi));
		taskQueue->push(taskPca);

		viewer->setImage(imagepca, roi);

		BackgroundTaskPtr task(new BackgroundTask(roi));
		QObject::connect(task.get(), SIGNAL(finished(bool)),
			this, SLOT(imgPcaCalculationComplete(bool)), Qt::QueuedConnection);
		taskQueue->push(task);

	}
		break;
	case GRADPCA:
	{
		// FIXME SAME SHIT AS ABOVE
		gradientpca->reset(new SharedMultiImgBase(new multi_img(0, 0, 0)));

		BackgroundTaskPtr taskPca(new MultiImg::PcaTbb(
			gradient, gradientpca, 0, roi));
		taskQueue->push(taskPca);

		viewer->setImage(gradientpca, roi);

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
	*/
}

void ViewerContainer::initUi()
{
	// Only one viewer per representation type is supported.
	foreach (representation::t i, representation::all())
		createViewer(i);

	// start with IMG, hide IMGPCA, GRADPCA at the beginning
	activeViewer = map.value(representation::IMG);
	activeViewer->setActive();
	map.value(representation::IMGPCA)->toggleFold();
	map.value(representation::GRADPCA)->toggleFold();

	// create layout and fill with viewers
	QVBoxLayout *vLayout = new QVBoxLayout(this);
	foreach(multi_img_viewer *viewer, map) {
		// add with stretch = 1 so they will stay evenly distributed in space
		vLayout->addWidget(viewer, 1);
	}
	vLayout->addStretch(); // align on top when all folded

	/* for self-activation of viewports
	 * representations are casted to their respective int */
	QSignalMapper *vpsmap = new QSignalMapper(this);
	foreach (representation::t i, representation::all()) {
		Viewport *viewport = map[i]->getViewport();
		vpsmap->setMapping(viewport, (int)i);
		connect(viewport, SIGNAL(activated()),
				vpsmap, SLOT(map()));
	}
	connect(vpsmap, SIGNAL(mapped(int)),
			this, SLOT(setActiveViewer(int)));

	foreach (representation::t i, representation::all()) {
		multi_img_viewer *viewer1 = map[i];
		Viewport *viewport1 = viewer1->getViewport();

		// connect pass through signals from markButton
		connect(this, SIGNAL(viewersToggleLabeled(bool)),
				viewer1, SLOT(toggleLabeled(bool)));
		// connect pass through signals from nonmarkButton
		connect(this, SIGNAL(viewersToggleUnlabeled(bool)),
				viewer1, SLOT(toggleUnlabeled(bool)));

		// todo make this globally consistent
		connect(viewer1, SIGNAL(finishTask(bool)),
				this, SLOT(finishTask(bool)));

		connect(viewer1, SIGNAL(finishedCalculation(representation::t)),
				this, SLOT(connectViewer(representation::t)));

		/* following stuff is used to pass to/from MainWindow / Controller */

		connect(viewport1, SIGNAL(bandSelected(representation::t, int)),
				this, SIGNAL(bandSelected(representation::t, int)));

		connect(viewer1, SIGNAL(setGUIEnabledRequested(bool,TaskType)),
				this, SIGNAL(setGUIEnabledRequested(bool,TaskType)));

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
		connect(viewer1, SIGNAL(toggleViewer(bool, representation::t)),
				this, SLOT(toggleViewer(bool, representation::t)));

		foreach (representation::t j, representation::all()) {
			multi_img_viewer *viewer2 = map[j];
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


multi_img_viewer *ViewerContainer::createViewer(representation::t type)
{
    multi_img_viewer *viewer = new multi_img_viewer(this);
	viewer->setSizePolicy(QSizePolicy::Preferred, // hor
						  QSizePolicy::Expanding); // ver
    viewer->setType(type);
	map.insert(type, viewer);
}

void ViewerContainer::newOverlay()
{
	emit drawOverlay(activeViewer->getHighlightMask());
}

void ViewerContainer::updateLabelsPartially(const cv::Mat1s &labels,
											const cv::Mat1b &mask)
{
	// is it worth it to do it incrementally (2 updates for each positive entry)
	bool profitable = ((2 * cv::countNonZero(mask)) < mask.total());
	if (profitable) {
		// gui disable
		emit setGUIEnabledRequested(false, TT_NONE);

		foreach (multi_img_viewer *i, map) {
			i->updateLabelsPartially(labels, mask);
		}

		// gui enable when done
		BackgroundTaskPtr taskEpilog(new BackgroundTask());
		QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)),
			this, SLOT(finishTask(bool)), Qt::QueuedConnection);
		taskQueue->push(taskEpilog);
	} else {
		// just update the whole thing
		updateLabels(labels);
	}
}
