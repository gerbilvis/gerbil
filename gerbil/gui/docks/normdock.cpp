#include "normdock.h"
#include "ui_normdock.h"

#include "../gerbil_gui_debug.h"

// sigh, still no min(), max() in C++
double min(double a, double b) {
	if(a<=b)
		return a;
	else
		return b;
}

double max(double a, double b) {
	if(a>=b)
		return a;
	else
		return b;
}

NormDock::NormDock(QWidget *parent) :
	QDockWidget(parent)
{
	// init with zero ranges
	ranges.insert(representation::IMG, ImageDataRange());
	ranges.insert(representation::GRAD, ImageDataRange());

	modes.insert(representation::IMG, MultiImg::NORM_FIXED);
	modes.insert(representation::GRAD, MultiImg::NORM_FIXED);

	setupUi(this);
	initUi();
}

NormDock::~NormDock()
{
}

void NormDock::initUi()
{

	normModeBox->addItem("Observed");
	normModeBox->addItem("Theoretical");
	normModeBox->addItem("Fixed");

	// FIXME: could not find implementation for clamping in ImageModel.
	// -> no Clamp button for now.
	normClampButton->hide();

	connect(normIButton, SIGNAL(toggled(bool)),
			this, SLOT(processNormTargetSelected()));
	connect(normGButton, SIGNAL(toggled(bool)),
			this, SLOT(processNormTargetSelected()));
	connect(normModeBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(processNormModeSelected(int)));
	connect(normMinBox, SIGNAL(editingFinished()),
			this, SLOT(processMinValueChanged()));
	connect(normMaxBox, SIGNAL(editingFinished()),
			this, SLOT(processMaxValueChanged()));
	connect(normApplyButton, SIGNAL(clicked()),
			this, SLOT(processApplyClicked()));

	// update values in GUI elements
	processNormTargetSelected();
}


void NormDock::setGuiEnabled(bool enable, TaskType tt)
{
	// original code from MainWindow:
	//	normDock->setEnabled((enable || tt == TT_NORM_RANGE || tt == TT_CLAMP_RANGE_IMG || tt == TT_CLAMP_RANGE_GRAD) && !limitedMode);
	//	normIButton->setEnabled(enable || tt == TT_NORM_RANGE || tt == TT_CLAMP_RANGE_IMG);
	//	normGButton->setEnabled(enable || tt == TT_NORM_RANGE || tt == TT_CLAMP_RANGE_GRAD);
	//	normModeBox->setEnabled(enable);
	//	normApplyButton->setEnabled(enable || tt == TT_NORM_RANGE);
	//	normClampButton->setEnabled(enable || tt == TT_CLAMP_RANGE_IMG || tt == TT_CLAMP_RANGE_GRAD);

	this->setEnabled( (enable ||
					   tt == TT_NORM_RANGE ||
					   tt == TT_CLAMP_RANGE_IMG ||
					   tt == TT_CLAMP_RANGE_GRAD
					  ) && !limitedMode);

	normIButton->setEnabled(enable ||
						 tt == TT_NORM_RANGE || tt == TT_CLAMP_RANGE_IMG);
	normGButton->setEnabled(enable ||
						 tt == TT_NORM_RANGE || tt == TT_CLAMP_RANGE_GRAD);
	normModeBox->setEnabled(enable);
	normApplyButton->setEnabled(enable || tt == TT_NORM_RANGE);
	normClampButton->setEnabled(enable ||
								tt == TT_CLAMP_RANGE_IMG || tt == TT_CLAMP_RANGE_GRAD);
}

void NormDock::setNormRange(representation::t type, const ImageDataRange &range)
{
	//GGDBGM(type << " " << range << endl);
	assert(representation::IMG == type || representation::GRAD == type );
	ranges[type] = range;
	// update GUI with new values
	processNormTargetSelected();
}

void NormDock::setNormMode(representation::t type,MultiImg::NormMode mode)
{
	assert(representation::IMG == type || representation::GRAD == type );
	modes[type] = mode;
	// update GUI with new values
	processNormTargetSelected();
}

void NormDock::setNormTarget(representation::t type)
{
	assert(representation::IMG == type || representation::GRAD == type );
	if(representation::IMG == type) {
		normIButton->toggle();
	} else { // GRAD
		normGButton->toggle();
	}
}


void NormDock::processApplyClicked()
{
	//GGDBG_CALL();
	emit normalizationParametersChanged(
				normTarget,
				modes[normTarget],
				ranges[normTarget]);
	emit applyNormalizationRequested();
}

void NormDock::processNormTargetSelected()
{
	if(normIButton->isChecked()) {
		normTarget = representation::IMG;
	} else {
		normTarget = representation::GRAD;
	}

	normModeBox->blockSignals(true);
	normModeBox->setCurrentIndex(static_cast<int>(modes[normTarget]));
	normModeBox->blockSignals(false);

	normMinBox->blockSignals(true);
	normMinBox->setValue(ranges[normTarget].min);
	normMinBox->blockSignals(false);

	normMaxBox->blockSignals(true);
	normMaxBox->setValue(ranges[normTarget].max);
	normMaxBox->blockSignals(false);
}

void NormDock::processMinValueChanged()
{
	//GGDBG_CALL();
	const double val = normMinBox->value();
	ranges[normTarget].min = val;
	//GGDBGM(ranges[normTarget].max << endl);
	ranges[normTarget].max = max(val, ranges[normTarget].max);
	//GGDBGM(ranges[normTarget].max << endl);
	processNormTargetSelected();
}

void NormDock::processMaxValueChanged()
{
	//GGDBG_CALL();
	const double val = normMaxBox->value();
	ranges[normTarget].max = val;
	ranges[normTarget].min = min(val, ranges[normTarget].min);
	processNormTargetSelected();
}


