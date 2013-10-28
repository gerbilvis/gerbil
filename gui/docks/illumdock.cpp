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

void IllumDock::initUi()
{
	for (int i = 0; i < 2; ++i) {
		QComboBox *b = (i ? illum2Box : illum1Box);
		b->addItem("Neutral", 0);
		b->addItem("2,856 K (Illuminant A, light bulb)",	2856);
		b->addItem("3,100 K (Tungsten halogen lamp)",		3100);
		b->addItem("5,000 K (Horizon light)",				5000);
		b->addItem("5,500 K (Mid-morning daylight)",		5500);
		b->addItem("6,500 K (Noon daylight)",				6500);
		b->addItem("7,500 K (North sky daylight)",			7500);
	}

	connect(illum1Box, SIGNAL(currentIndexChanged(int)),
			this, SLOT(onIllum1Selected(int)));
	connect(illum2Box, SIGNAL(currentIndexChanged(int)),
			this, SLOT(onIllum2Selected(int)));
	/* This indicates if the user wants the illumination curve
	/* to be shown in the viewer. */
	connect(showCheck, SIGNAL(toggled(bool)),
			this, SLOT(onShowToggled(bool)));
	showCheck->setVisible(false);
	connect(applyButton, SIGNAL(clicked()),
			this, SLOT(onApplyClicked()));
}


void IllumDock::onIllum1Selected(int idx)
{
	// i1: Temp. in Kelvin
	int i1 = illum1Box->itemData(idx).value<int>();
	showCheck->setEnabled(i1 > 0);
	showCheck->setVisible(i1 > 0);
	emit illum1Selected(i1);
}

void IllumDock::onIllum2Selected(int idx)
{
	// i2: Temp. in Kelvin
	int i2 = illum1Box->itemData(idx).value<int>();
	emit illum2Selected(i2);
}

void IllumDock::onShowToggled(bool show)
{
	emit showIlluminationCurveChanged(show);
}

void IllumDock::onApplyClicked()
{
	int i1 = illum1Box->itemData(illum1Box->currentIndex()).value<int>();
	int i2 = illum2Box->itemData(illum2Box->currentIndex()).value<int>();
	if (i1 == i2)
		return;

	illum1Box->setDisabled(true);
	showCheck->setVisible(true);
	showCheck->setChecked(true);

	emit applyIllum();

	/* reflect change in our own gui, will propagate accordingly */
	illum1Box->setCurrentIndex(illum2Box->currentIndex());
}
