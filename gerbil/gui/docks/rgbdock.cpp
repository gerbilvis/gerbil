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

void RgbDock::updatePixmap(coloring type, bool gradient, QPixmap p)
{
	//GGDBG_CALL();
	// only use selected (false-)coloring
	if (type != displayType || gradient != displayGradient)
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
	sourceBox->addItem("Color Matching Functions", CMF);
	sourceBox->addItem("Principle Component Analysis", PCA);
#ifdef WITH_EDGE_DETECT
	sourceBox->addItem("Self-organizing Map", SOM);
#endif // WITH_EDGE_DETECT
	sourceBox->setCurrentIndex(0);

	connect(sourceBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(selectColorRepresentation()));

	connect(applyButton, SIGNAL(clicked()),
			this, SLOT(calculateColorRepresentation()));

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
		emit rgbRequested(displayType, displayGradient);
	}
}

void RgbDock::processImageUpdate(representation::t, SharedMultiImgPtr)
{
	//GGDBGM(format("visible=%1%  rgbValid=%2%")%dockVisible %rgbValid <<endl);
	rgbValid = false;
	view->setEnabled(false);
	if (dockVisible) {
		//GGDBGM("requesting rgb"<<endl);
		emit rgbRequested(displayType, displayGradient);
	}
}

void RgbDock::selectColorRepresentation()
{
	QVariant boxData = sourceBox->itemData(sourceBox->currentIndex());
	displayType = (FalseColorModel::coloring)boxData.toInt();
	// we do not need rest of state to ask for lazy update

	/* kindly ask if we could have the image without effort right now:
	 * no re-calculation
	 */
	emit rgbLazyRequested(displayType, displayGradient);
}

void RgbDock::calculateColorRepresentation()
{
	// apply selected state (display type update was triggered by combobox)
	displayGradient = gradientCheck->isChecked();

	rgbValid = false;
	view->setEnabled(false);
	emit rgbRequested(displayType, displayGradient);
}
