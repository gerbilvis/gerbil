#include "viewercontainer.h"
//#include "ui_viewerswidget.h"

#include "viewport.h"
#include "mainwindow.h"

#include <QSignalMapper>

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

void ViewerContainer::addImage(representation repr, sets_ptr temp,
                               const std::vector<cv::Rect> &regions,
                               cv::Rect roi)
{
	multi_img_viewer *viewer = vm.value(repr);
	viewer->addImage(temp, regions, roi);
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

void ViewerContainer::finishViewerRefresh(int viewer)
{
	// TODO impl
}

void ViewerContainer::finishTask(bool success)
{
	if(success)
		emit requestGUIEnabled(true, TT_NONE);
}

void ViewerContainer::toggleViewerEnable(representation repr)
{
	// FIXME handle state in viewer
	// disconnectViewer(viewer);

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

    // CAVEAT: Only one viewer per representation type is supported now.
	// While there is basic support for multpile viewers per representation,
	// in many places one viewer per representation is still assumed.
    //
    createViewer(IMG);
    createViewer(GRAD);
    createViewer(IMGPCA);
    createViewer(GRADPCA);

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
			this, SLOT(setviewportActive(int)));

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
		connect(this, SIGNAL(viewersUnToggleLabeled(bool)),
				viewer1, SLOT(toggleUnLabeled(bool)));

		connect(viewport1, SIGNAL(bandSelected(representation, int)),
				this, SIGNAL(viewportBandSelected(representation,int)));

		connect(viewer1, SIGNAL(setGUIEnabled(bool, TaskType)),
				this, SIGNAL(viewerSetGUIEnabled(bool, TaskType)));
		connect(viewer1, SIGNAL(finishTask(bool)),
				this, SIGNAL(viewerFinishTask(bool)));

		connect(viewer1, SIGNAL(newOverlay()),
				this, SIGNAL(viewerNewOverlay()));
		connect(viewport1, SIGNAL(newOverlay(int)),
				this, SIGNAL(viewportNewOverlay()));

		connect(viewport1, SIGNAL(addSelection()),
				this, SIGNAL(viewportAddToLabel()));
		connect(viewport1, SIGNAL(remSelection()),
				this, SIGNAL(viewportRemFromLabel()));

		// non-pass-through
		connect(viewer1, SIGNAL(toggleViewer(bool , representation)),
				this, SLOT(toggleViewer(bool , representation)));

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

