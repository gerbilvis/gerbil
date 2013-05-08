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

void ViewerContainer::addImage(representation repr, sets_ptr temp,
                               const std::vector<cv::Rect> &regions,
                               cv::Rect roi)
{
    ViewerList viewers;
    viewers = vm.values(repr);
    foreach (multi_img_viewer *v, viewers) {
        v->addImage(temp, regions, roi);
    }
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

    multi_img_viewer *viewer;
    viewer = createViewer(IMG);
    viewer = createViewer(GRAD);
    viewer = createViewer(IMGPCA);
    viewer = createViewer(GRADPCA);

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
		connect(viewer1, SIGNAL(toggleViewer(bool , representation)),
				this, SIGNAL(viewerToggleViewer(bool , representation)));
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
    vl.append(viewer);
    vm.insert(repr, viewer);
    vLayout->addWidget(viewer);
}
