#include <QVBoxLayout>

#include <iostream>
#include "gerbil_gui_debug.h"
#include "widgets/scaledview.h"
#include "rgbdock.h"
#include "../model/falsecolormodel.h"

RgbDock::RgbDock(QWidget *parent) :
	QDockWidget(parent), displayType(CMF), displayGradient(0)
{
	setupUi(this);
	initUi();

	gradientCheck->setEnabled(displayType != CMF); // CMF does not work on gradient
}

void RgbDock::updatePixmap(coloring type, bool gradient, QPixmap p)
{
	//GGDBG_CALL();
	// only use selected (false-)coloring
	if (type != displayType || gradient != currGradient())
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

	connect(gradientCheck, SIGNAL(stateChanged(int)),
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
		emit rgbRequested(displayType, currGradient(), false);
	}
}

void RgbDock::processImageUpdate(representation::t, SharedMultiImgPtr)
{
	//GGDBGM(format("visible=%1%  rgbValid=%2%")%dockVisible %rgbValid <<endl);
	rgbValid = false;
	view->setEnabled(false);
	if (dockVisible) {
		//GGDBGM("requesting rgb"<<endl);
		emit rgbRequested(displayType, currGradient(), false);
	}
}

void RgbDock::selectColorRepresentation()
{
	QVariant boxData = sourceBox->itemData(sourceBox->currentIndex());
	displayType = (FalseColorModel::coloring)boxData.toInt();
	displayGradient = gradientCheck->isChecked();

	gradientCheck->setEnabled(displayType != CMF); // CMF does not work on gradient

	/* kindly ask if we could have the image without effort right now:
	 * no re-calculation
	 */
	emit rgbLazyRequested(displayType, currGradient());
}

void RgbDock::calculateColorRepresentation()
{
	// apply selected state (display type update was triggered by combobox)
	displayGradient = gradientCheck->isChecked();

	rgbValid = false;
	view->setEnabled(false);
	emit rgbRequested(displayType, currGradient(), true);
}
