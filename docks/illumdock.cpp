#include "illumdock.h"

IllumDock::IllumDock(QWidget *parent) :
	QDockWidget(parent), Ui::IllumDock()
{
	setupUi(this);
	initUi();
}

IllumDock::~IllumDock()
{
}

void IllumDock::onApplyClicked()
{
	int i1 = i1Box->itemData(i1Box->currentIndex()).value<int>();
	int i2 = i2Box->itemData(i2Box->currentIndex()).value<int>();
	if (i1 == i2)
		return;

	i1Box->setDisabled(true);
	i1Check->setVisible(true);

	emit applyIllum();

	/* reflect change in our own gui (will propagate to IMG viewer) */
	i1Box->setCurrentIndex(i2Box->currentIndex());
}


void IllumDock::initUi()
{
	for (int i = 0; i < 2; ++i) {
		QComboBox *b = (i ? i2Box : i1Box);
		b->addItem("Neutral", 0);
		b->addItem("2,856 K (Illuminant A, light bulb)",	2856);
		b->addItem("3,100 K (Tungsten halogen lamp)",		3100);
		b->addItem("5,000 K (Horizon light)",				5000);
		b->addItem("5,500 K (Mid-morning daylight)",		5500);
		b->addItem("6,500 K (Noon daylight)",				6500);
		b->addItem("7,500 K (North sky daylight)",			7500);
	}

	connect(i1Box, SIGNAL(currentIndexChanged(int)),
			this, SLOT(onIllum1Selected(int)));
	connect(i2Box, SIGNAL(currentIndexChanged(int)),
			this, SLOT(onIllum2Selected(int)));
	/* This indicates if the user wants the illumination curve
	/* to be shown in the viewer. */
	connect(i1Check, SIGNAL(toggled(bool)),
			this, SLOT(onShowToggled(bool)));
	i1Check->setVisible(false);
	connect(i2Button, SIGNAL(clicked()),
			this, SLOT(onApplyClicked()));
}


void IllumDock::onIllum1Selected(int idx)
{
	// i1: Temp. in Kelvin
	int i1 = i1Box->itemData(idx).value<int>();
	i1Check->setEnabled(i1 > 0);
	emit illum1Selected(i1);
	emit showIlluminationCurveChanged(i1 > 0);
}

void IllumDock::onIllum2Selected(int idx)
{
	// i2: Temp. in Kelvin
	int i2 = i1Box->itemData(idx).value<int>();
	emit illum2Selected(i2);
	if(0==i2) {
		emit showIlluminationCurveChanged(false);
	}
}

void IllumDock::onShowToggled(bool show)
{
	emit showIlluminationCurveChanged(show);
}
