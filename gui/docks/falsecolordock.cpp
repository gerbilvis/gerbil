#include <QVBoxLayout>
#include <QThread>

#include <iostream>
#include "gerbil_gui_debug.h"
#include "falsecolordock.h"
#include "../model/falsecolormodel.h"
#include "../widgets/scaledview.h"
#include "../widgets/autohideview.h"
#include "../widgets/autohidewidget.h"

std::ostream &operator<<(std::ostream& os, const FalseColoringState::Type& state)
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
		<< "True Color (CIE XYZ)"
		<< "Principle Component Analysis (PCA)"
		<< "Spectral-gradient PCA"
		<< "Self-organizing Map (SOM)"
		<< "Spectral-gradient SOM";

FalseColorDock::FalseColorDock(QWidget *parent)
	: QDockWidget(parent), lastShown(FalseColoring::CMF)
{
	/* setup our UI here as it is quite minimalistic */
	QWidget *contents = new QWidget(this);
	QVBoxLayout *layout = new QVBoxLayout(contents);
	view = new AutohideView(contents);
	view->setBaseSize(QSize(250, 300));
	view->setFrameShape(QFrame::NoFrame);
	view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	layout->addWidget(view);
	this->setWidget(contents);

	initUi();
}

void FalseColorDock::processColoringComputed(FalseColoring::Type coloringType, QPixmap p)
{
//	GGDBGM("enterState():"<<endl);
	enterState(coloringType, FalseColoringState::FINISHED);
	coloringUpToDate[coloringType] = true;
	updateTheButton();
	updateProgressBar();
	if (coloringType == selectedColoring()) {
		view->setEnabled(true);
		scene->setPixmap(p);
		view->update();
		view->setToolTip(prettyFalseColorNames[coloringType]);
		this->setWindowTitle(prettyFalseColorNames[coloringType]);
		lastShown = coloringType;
	}
}

void FalseColorDock::processComputationCancelled(FalseColoring::Type coloringType)
{
	if(coloringState[coloringType] == FalseColoringState::ABORTING) {
		coloringProgress[coloringType] = 0;
//		GGDBGM("enterState():"<<endl);
		enterState(coloringType, FalseColoringState::FINISHED);
		updateTheButton();
		updateProgressBar();
	} else if(coloringState[coloringType] == FalseColoringState::CALCULATING) {
//		GGDBGM("restarting cancelled computation"<<endl);
		requestColoring(coloringType);
	}
}

void FalseColorDock::processSelectedColoring()
{
	//GGDBGM( "requesting false color image " << selectedColoring() << endl);
	requestColoring(selectedColoring());
	updateTheButton();
	updateProgressBar();
}

void FalseColorDock::processApplyClicked()
{
	if(coloringState[selectedColoring()] == FalseColoringState::CALCULATING) {
//		GGDBGM("enterState():"<<endl);
		enterState(selectedColoring(), FalseColoringState::ABORTING);
		emit cancelComputationRequested(selectedColoring());
		// go back to last shown coloring
		if(coloringUpToDate[lastShown]) {
			uisel->sourceBox->setCurrentIndex(int(lastShown));
			requestColoring(lastShown);
		} else { // or fall back to CMF, e.g. after ROI change
			uisel->sourceBox->setCurrentIndex(FalseColoring::CMF);
			requestColoring(FalseColoring::CMF);
		}
	} else if(coloringState[selectedColoring()] == FalseColoringState::FINISHED) {
		requestColoring(selectedColoring(), /* recalc */ true);
	}
}

void FalseColorDock::initUi()
{
	// initialize scaled view
	view->init();
	scene = new ScaledView();
	view->setScene(scene);
	connect(scene, SIGNAL(newContentRect(QRect)),
			view, SLOT(fitContentRect(QRect)));

	// initialize selection widget
	sel = new AutohideWidget();
	uisel = new Ui::FalsecolorDockSelUI();
	uisel->setupUi(sel);
	scene->offTop = AutohideWidget::OutOffset;
	view->addWidget(AutohideWidget::TOP, sel);

	// fill up source choices
	uisel->sourceBox->addItem(prettyFalseColorNames[FalseColoring::CMF],
					   FalseColoring::CMF);
	uisel->sourceBox->addItem(prettyFalseColorNames[FalseColoring::PCA],
					   FalseColoring::PCA);
	uisel->sourceBox->addItem(prettyFalseColorNames[FalseColoring::PCAGRAD],
					   FalseColoring::PCAGRAD);
#ifdef WITH_EDGE_DETECT
	uisel->sourceBox->addItem(prettyFalseColorNames[FalseColoring::SOM],
					   FalseColoring::SOM);
	uisel->sourceBox->addItem(prettyFalseColorNames[FalseColoring::SOMGRAD],
					   FalseColoring::SOMGRAD);
#endif // WITH_EDGE_DETECT
	uisel->sourceBox->setCurrentIndex(0);

	updateTheButton();
	updateProgressBar();

	connect(scene, SIGNAL(newSizeHint(QSize)),
			view, SLOT(updateSizeHint(QSize)));

	connect(uisel->sourceBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(processSelectedColoring()));

	connect(uisel->theButton, SIGNAL(clicked()),
			this, SLOT(processApplyClicked()));

	connect(this, SIGNAL(visibilityChanged(bool)),
			this, SLOT(processVisibilityChanged(bool)));
}

FalseColoring::Type FalseColorDock::selectedColoring()
{
	QComboBox *src = uisel->sourceBox;
	QVariant boxData = src->itemData(src->currentIndex());
	FalseColoring::Type coloringType = FalseColoring::Type(boxData.toInt());
	return coloringType;
}

void FalseColorDock::requestColoring(FalseColoring::Type coloringType, bool recalc)
{
//	GGDBGM("enterState():"<<endl);
	enterState(coloringType, FalseColoringState::CALCULATING);
	updateTheButton();
	emit falseColoringRequested(coloringType, recalc);
}

void FalseColorDock::updateProgressBar()
{
	if (coloringState[selectedColoring()] == FalseColoringState::CALCULATING) {
		int percent = coloringProgress[selectedColoring()];
		uisel->calcProgress->setValue(percent);
		uisel->calcProgress->setVisible(true);
		sel->adjust(); // grow accordingly
		// stay visible
		sel->scrollIn(true);
	} else {
		std::cerr << prettyFalseColorNames[selectedColoring()].toStdString()
				<< " not CALCULATING" << std::endl;
		uisel->calcProgress->setValue(0);
		uisel->calcProgress->setVisible(false);
		sel->adjust(); // shrink accordingly
		// remove enforced visibility
		sel->scrollOut();
	}
}

void FalseColorDock::updateTheButton()
{
	switch (coloringState[selectedColoring()]) {
	case FalseColoringState::FINISHED:
		uisel->theButton->setIcon(QIcon::fromTheme("view-refresh"));
		uisel->theButton->setText("Recalculate");
		uisel->theButton->setVisible(false);
		if( selectedColoring()==FalseColoring::SOM ||
			selectedColoring()==FalseColoring::SOMGRAD)
		{
			uisel->theButton->setVisible(true);
		}
		break;
	case FalseColoringState::CALCULATING:
		uisel->theButton->setIcon(QIcon::fromTheme("process-stop"));
		uisel->theButton->setText("Cancel");
		uisel->theButton->setVisible(true);
		break;
	case FalseColoringState::ABORTING:
		uisel->theButton->setVisible(true);
		break;
	default:
		assert(false);
		break;
	}
}

void FalseColorDock::enterState(FalseColoring::Type coloringType, FalseColoringState::Type state)
{
//	GGDBGM(coloringType << " entering state " << state << endl);
	coloringState[coloringType] = state;
}

void FalseColorDock::processVisibilityChanged(bool visible)
{
	dockVisible = visible;
	if(dockVisible) {
		requestColoring(selectedColoring());
	}
}

void FalseColorDock::processColoringOutOfDate(FalseColoring::Type coloringType)
{
	coloringUpToDate[coloringType] = false;
	if(dockVisible) {
		requestColoring(selectedColoring());
	}
}

void FalseColorDock::processCalculationProgressChanged(FalseColoring::Type coloringType, int percent)
{
	coloringProgress[coloringType] = percent;

	if (coloringType == selectedColoring())
		updateProgressBar();
}
