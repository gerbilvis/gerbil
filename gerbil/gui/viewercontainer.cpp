#include "viewercontainer.h"
//#include "ui_viewerswidget.h"

ViewerContainer::ViewerContainer(QWidget *parent) :
    QWidget(parent)
    //ui(new Ui::ViewerContainer)
{
    //ui->setupUi(this);
    initUi();
}

ViewerContainer::~ViewerContainer()
{
    //delete ui;
}

void ViewerContainer::initUi()
{
    vLayout = new QVBoxLayout(this);

    multi_img_viewer *viewer;
    viewer = createViewer(IMG);
    viewer = createViewer(GRAD);
    viewer = createViewer(IMGPCA);
    viewer = createViewer(GRADPCA);
}

multi_img_viewer *ViewerContainer::createViewer(representation rep)
{
    multi_img_viewer *viewer = new multi_img_viewer(this);
    vl.append(viewer);
    vm.insert(rep, viewer);
    vLayout->addWidget(viewer);
}
