#include <QVBoxLayout>
#include <QThread>

#include <iostream>
#include "gerbil_gui_debug.h"
#include "widgets/scaledview.h"
#include "rgbdock.h"
#include "../model/falsecolormodel.h"

RgbDock::RgbDock(QWidget *parent)
	: QDockWidget(parent)
{
	setupUi(this);
	initUi();
}

void RgbDock::processColoringComputed(FalseColoring::Type coloringType, QPixmap p)
{
	//GGDBG_CALL();
	coloringState[coloringType] = RgbDockState::FINISHED;
	updateProgressBar();
	if (coloringType == selectedColoring()) {
		theButton->setText("Re-Calculate");
		view->setEnabled(true);
		view->setPixmap(p);
		view->update();
	}
}

void RgbDock::processComputationCancelled(FalseColoring::Type coloringType)
{
	//GGDBG_CALL();
	calcProgress->setVisible(false);
	theButton->setText("Re-Calculate");
	coloringProgress[coloringType] = 0;
	coloringState[coloringType] = RgbDockState::FINISHED;
	updateProgressBar();
}

void RgbDock::processSelectedColoring()
{
	//GGDBGM( "requesting false color image " << selectedColoring() << endl);
	requestColoring(selectedColoring());
	if(!(selectedColoring()==FalseColoring::SOM ||
		 selectedColoring()==FalseColoring::SOMGRAD ))
	{
		theButton->setVisible(false);
	} else {
		theButton->setVisible(true);
	}
	updateProgressBar();
}

void RgbDock::processApplyClicked()
{
	if(coloringState[selectedColoring()] == RgbDockState::CALCULATING) {
		emit cancelComputationRequested(selectedColoring());
	} else if(coloringState[selectedColoring()] == RgbDockState::FINISHED) {
		requestColoring(selectedColoring(), /* recalc */ true);
	}
}

void RgbDock::debugProgressValue(int v)
{
	//GGDBGM("value " << v << endl);
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

	calcProgress->setValue(0);
	calcProgress->setVisible(false);

	theButton->setVisible(false);

	connect(calcProgress, SIGNAL(valueChanged(int)),
			this, SLOT(debugProgressValue(int)));

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
	theButton->setText("Cancel Computation");
	coloringState[coloringType] = RgbDockState::CALCULATING;
	emit falseColoringRequested(coloringType, recalc);
}

void RgbDock::requestCancelComputation(FalseColoring::Type coloringType)
{
	emit cancelComputationRequested(coloringType);
}

void RgbDock::updateProgressBar()
{
	//GGDBGM("thread" << QThread::currentThread () << endl);
	if(coloringState[selectedColoring()] == RgbDockState::CALCULATING) {
		int percent = coloringProgress[selectedColoring()];
		// FIXME for some reason progressBar displays 100% even if we set 0% here...
		// ?!?
//		GGDBGM(percent << "%"<< endl);
		if(!calcProgress->isVisible()) {
			calcProgress->setVisible(true);
		}
		calcProgress->setValue(percent);
		calcProgress->update();
		update();
	} else {
		//GGDBGM("hiding"<<endl);
		calcProgress->setVisible(false);
	}
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
