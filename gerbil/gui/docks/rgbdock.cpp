#include <QVBoxLayout>

#include <iostream>
#include "gerbil_gui_debug.h"
#include "../scaledview.h"
#include "rgbdock.h"
#include "../model/falsecolor.h"

RgbDock::RgbDock(QWidget *parent) :
	QDockWidget(parent), displayType(CMF)
{
	setupUi(this);
	initUi();
}

void RgbDock::updatePixmap(coloring type, QPixmap p)
{
	//GGDBG_CALL();
	// only use selected (false-)coloring
	if (type != displayType)
		return;

	view->setEnabled(true);
	view->setPixmap(p);
	view->update();
	rgbValid=true;

	/* TODO: old(from johannes, not sure what this is to mean):
	 * We could think about data-sharing between image model
	 * and falsecolor model for the CMF part.
	 */
}

void RgbDock::initUi()
{
//	setWindowTitle("RGB View");
//	QWidget *child = new QWidget(this);
//	QVBoxLayout *layout = new QVBoxLayout();
//	view = new ScaledView();
//	view->setMinimumSize(150,150);
//	layout->addWidget(view);
//	child->setLayout(layout);
//	child->setSizePolicy(
//				QSizePolicy::MinimumExpanding,
//				QSizePolicy::MinimumExpanding);
//	setWidget(child);
//	view->setEnabled(false);

	sourceBox->addItem("Color Matching Functions", CMF);
	sourceBox->addItem("Principle Component Analysis", PCA);
#ifndef WITH_EDGE_DETECT
	sourceBox->addItem("Self-organizing Map", SOM);
#endif // WITH_EDGE_DETECT
	sourceBox->setCurrentIndex(0);

	connect(applyButton, SIGNAL(clicked()),
			this, SLOT(changeColorRepresentation()));

	connect(this, SIGNAL(visibilityChanged(bool)),
			this, SLOT(processVisibilityChanged(bool)));
}

void RgbDock::processVisibilityChanged(bool visible)
{
	dockVisible = visible;
	//GGDBGM(format("visible=%1%  rgbValid=%2%")%dockVisible %rgbValid <<endl);
	if(dockVisible && !rgbValid) {
		//GGDBGM("requesting rgb"<<endl);
		view->setEnabled(false);
		emit rgbRequested(displayType);
	}
}

void RgbDock::processImageUpdate(representation::t, SharedMultiImgPtr)
{
	//GGDBGM(format("visible=%1%  rgbValid=%2%")%dockVisible %rgbValid <<endl);
	rgbValid = false;
	view->setEnabled(false);
	if (dockVisible) {
		//GGDBGM("requesting rgb"<<endl);
		emit rgbRequested(displayType);
	}
}

void RgbDock::changeColorRepresentation()
{
	// apply selected state
	QVariant boxData = sourceBox->itemData(sourceBox->currentIndex());
	displayType = (FalseColorModel::coloring)boxData.toInt();
	displayGradient = gradientCheck->isChecked();

	rgbValid = false;
	view->setEnabled(false);
	emit rgbRequested(displayType);
}
