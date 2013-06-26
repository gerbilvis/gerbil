#include "banddock.h"
#include "ui_banddock.h"

#include "../gerbil_gui_debug.h"

BandDock::BandDock(QWidget *parent) :
	QDockWidget(parent)
{
	setupUi(this);
	initUi();
}

BandDock::~BandDock()
{
}

void BandDock::initUi()
{
	connect(&bv->labelTimer, SIGNAL(timeout()),
			bv, SLOT(commitLabelChanges()));

	connect(markerSelector, SIGNAL(currentIndexChanged(int)),
			bv, SLOT(changeCurrentLabel(int)));

	connect(alphaSlider, SIGNAL(valueChanged(int)),
			bv, SLOT(applyLabelAlpha(int)));


	// -> DockController
	//	connect(bandView, SIGNAL(alteredLabels(const cv::Mat1s&, const cv::Mat1b&)),
	//			this, SIGNAL(alterLabelingRequested(cv::Mat1s,cv::Mat1b)));
	//	connect(bandView, SIGNAL(newLabeling(const cv::Mat1s&)),
	//			this, SIGNAL(newLabelingRequested(cv::Mat1s)));

	/* when applybutton is pressed, bandView commits full label matrix */
	connect(applyButton, SIGNAL(clicked()),
			bv, SLOT(commitLabels()));
}

void BandDock::changeBand(QPixmap band, QString desc)
{
	GGDBGM(band.width()<<endl);

	bv->setEnabled(true);
	bv->setPixmap(band);
	setWindowTitle(desc);
}
