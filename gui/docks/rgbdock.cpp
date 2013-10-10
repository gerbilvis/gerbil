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

static QStringList prettyFalseColorNames = QStringList()
		<< "Color Matching Functions"
		<< "Color Matching Functions on gradient"
		<< "Principle Component Analysis"
		<< "Principle Component Analysis on gradient"
		<< "Self-organizing Map"
		<< "Self-organizing Map on gradient";

RgbDock::RgbDock(QWidget *parent)
	: QDockWidget(parent), lastShown(FalseColoring::CMF)
{
	setupUi(this);
	initUi();
}

void RgbDock::processColoringComputed(FalseColoring::Type coloringType, QPixmap p)
{
//	GGDBGM("enterState():"<<endl);
	enterState(coloringType, RgbDockState::FINISHED);
	coloringUpToDate[coloringType] = true;
	updateTheButton();
	updateProgressBar();
	if (coloringType == selectedColoring()) {
		view->setEnabled(true);
		view->setPixmap(p);
		view->update();
		view->setToolTip(prettyFalseColorNames[coloringType]);
		lastShown = coloringType;
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
		// go back to last shown coloring
		if(coloringUpToDate[lastShown]) {
			sourceBox->setCurrentIndex(int(lastShown));
			requestColoring(lastShown);
		} else { // or fall back to CMF, e.g. after ROI change
			sourceBox->setCurrentIndex(FalseColoring::CMF);
			requestColoring(FalseColoring::CMF);
		}
	} else if(coloringState[selectedColoring()] == RgbDockState::FINISHED) {
		requestColoring(selectedColoring(), /* recalc */ true);
	}
}

void RgbDock::initUi()
{
	sourceBox->addItem(prettyFalseColorNames[FalseColoring::CMF],
					   FalseColoring::CMF);
	sourceBox->addItem(prettyFalseColorNames[FalseColoring::CMFGRAD],
					   FalseColoring::CMFGRAD);
	sourceBox->addItem(prettyFalseColorNames[FalseColoring::PCA],
					   FalseColoring::PCA);
	sourceBox->addItem(prettyFalseColorNames[FalseColoring::PCAGRAD],
					   FalseColoring::PCAGRAD);
#ifdef WITH_EDGE_DETECT
	sourceBox->addItem(prettyFalseColorNames[FalseColoring::SOM],
					   FalseColoring::SOM);
	sourceBox->addItem(prettyFalseColorNames[FalseColoring::SOMGRAD],
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
	coloringUpToDate[coloringType] = false;
	if(dockVisible) {
		requestColoring(selectedColoring());
	}
}

void RgbDock::processCalculationProgressChanged(FalseColoring::Type coloringType, int percent)
{
	coloringProgress[coloringType] = percent;
	updateProgressBar();
}
