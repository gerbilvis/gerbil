#include <QVBoxLayout>
#include <QThread>

#include <iostream>
#include "gerbil_gui_debug.h"
#include "widgets/scaledview.h"
#include "rgbdock.h"
#include "../model/falsecolormodel.h"


std::ostream &operator<<(std::ostream& os, const RgbDockState::Type& state)
{
	if (state < 0 || state >= 3) {
		os << "INVALID";
		return os;
	}
	const char * const str[] = { "FINISHED", "CALCULATING", "ABORTING"};
	os << str[state];
	return os;
}


RgbDock::RgbDock(QWidget *parent)
	: QDockWidget(parent)
{
	setupUi(this);
	initUi();
}

void RgbDock::processColoringComputed(FalseColoring::Type coloringType, QPixmap p)
{
//	GGDBGM("enterState():"<<endl);
	enterState(coloringType, RgbDockState::FINISHED);
	updateTheButton();
	updateProgressBar();
	if (coloringType == selectedColoring()) {
		view->setEnabled(true);
		view->setPixmap(p);
		view->update();
	}
}

void RgbDock::processComputationCancelled(FalseColoring::Type coloringType)
{
	if(coloringState[coloringType] == RgbDockState::ABORTING) {
		coloringProgress[coloringType] = 0;
//		GGDBGM("enterState():"<<endl);
		enterState(coloringType, RgbDockState::FINISHED);
		updateTheButton();
		updateProgressBar();
	} else if(coloringState[coloringType] == RgbDockState::CALCULATING) {
//		GGDBGM("restarting cancelled computation"<<endl);
		requestColoring(coloringType);
	}
}

void RgbDock::processSelectedColoring()
{
	//GGDBGM( "requesting false color image " << selectedColoring() << endl);
	requestColoring(selectedColoring());
	updateTheButton();
	updateProgressBar();
}

void RgbDock::processApplyClicked()
{
	if(coloringState[selectedColoring()] == RgbDockState::CALCULATING) {
//		GGDBGM("enterState():"<<endl);
		enterState(selectedColoring(), RgbDockState::ABORTING);
		emit cancelComputationRequested(selectedColoring());
	} else if(coloringState[selectedColoring()] == RgbDockState::FINISHED) {
		requestColoring(selectedColoring(), /* recalc */ true);
	}
}

void RgbDock::initUi()
{
	// TODO: add tooltip
	sourceBox->addItem("Color Matching Functions",
					   FalseColoring::CMF);
	sourceBox->addItem("Color Matching Functions on gradient",
					   FalseColoring::CMFGRAD);
	sourceBox->addItem("Principle Component Analysis",
					   FalseColoring::PCA);
	sourceBox->addItem("Principle Component Analysis on gradient",
					   FalseColoring::PCAGRAD);
#ifdef WITH_EDGE_DETECT
	// TODO: add tooltip
	sourceBox->addItem("Self-organizing Map",
					   FalseColoring::SOM);
	sourceBox->addItem("Self-organizing Map on gradient",
					   FalseColoring::SOMGRAD);
#endif // WITH_EDGE_DETECT
	sourceBox->setCurrentIndex(0);

	updateTheButton();
	updateProgressBar();


	connect(sourceBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(processSelectedColoring()));

	connect(theButton, SIGNAL(clicked()),
			this, SLOT(processApplyClicked()));


	connect(this, SIGNAL(visibilityChanged(bool)),
			this, SLOT(processVisibilityChanged(bool)));
}

FalseColoring::Type RgbDock::selectedColoring()
{
	QVariant boxData = sourceBox->itemData(sourceBox->currentIndex());
	FalseColoring::Type coloringType = FalseColoring::Type(boxData.toInt());
	return coloringType;
}

void RgbDock::requestColoring(FalseColoring::Type coloringType, bool recalc)
{
//	GGDBGM("enterState():"<<endl);
	enterState(coloringType, RgbDockState::CALCULATING);
	updateTheButton();
	emit falseColoringRequested(coloringType, recalc);
}

void RgbDock::updateProgressBar()
{
	if(coloringState[selectedColoring()] == RgbDockState::CALCULATING) {
		int percent = coloringProgress[selectedColoring()];
		calcProgress->setVisible(true);
		calcProgress->setValue(percent);
	} else {
		calcProgress->setValue(0);
		calcProgress->setVisible(false);
	}
}

void RgbDock::updateTheButton()
{
	switch (coloringState[selectedColoring()]) {
	case RgbDockState::FINISHED:
		theButton->setText("Re-Calculate");
		theButton->setVisible(false);
		if( selectedColoring()==FalseColoring::SOM ||
			selectedColoring()==FalseColoring::SOMGRAD)
		{
			theButton->setVisible(true);
		}
		break;
	case RgbDockState::CALCULATING:
		theButton->setText("Cancel Computation");
		theButton->setVisible(true);
		break;
	case RgbDockState::ABORTING:
		theButton->setVisible(true);
		break;
	default:
		assert(false);
		break;
	}
}

void RgbDock::enterState(FalseColoring::Type coloringType, RgbDockState::Type state)
{
//	GGDBGM(coloringType << " entering state " << state << endl);
	coloringState[coloringType] = state;
}

void RgbDock::processVisibilityChanged(bool visible)
{
	// do we get this when first shown? -> Yes
	dockVisible = visible;
	//GGDBGM(format("visible=%1%  rgbValid=%2%")%dockVisible %rgbValid <<endl);
	if(dockVisible) {
		requestColoring(selectedColoring());
	}
}

void RgbDock::processColoringOutOfDate(FalseColoring::Type coloringType)
{
	if(dockVisible) {
		requestColoring(selectedColoring());
	}
}

void RgbDock::processCalculationProgressChanged(FalseColoring::Type coloringType, int percent)
{
	coloringProgress[coloringType] = percent;
	updateProgressBar();
}
