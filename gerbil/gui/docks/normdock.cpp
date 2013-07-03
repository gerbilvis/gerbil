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

	modes.insert(representation::IMG, MultiImg::NORM_OBSERVED);
	modes.insert(representation::GRAD, MultiImg::NORM_OBSERVED);

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
			this, SLOT(updateGUI()));
	connect(normGButton, SIGNAL(toggled(bool)),
			this, SLOT(updateGUI()));
	connect(normModeBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(processNormModeSelected(int)));
	connect(normMinBox, SIGNAL(editingFinished()),
			this, SLOT(processMinValueChanged()));
	connect(normMaxBox, SIGNAL(editingFinished()),
			this, SLOT(processMaxValueChanged()));
	connect(normApplyButton, SIGNAL(clicked()),
			this, SLOT(processApplyClicked()));

	// update values in GUI elements
	updateGUI();
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
	if(!(representation::IMG == type || representation::GRAD == type ))
		return;
	ranges[type] = range;
	// update GUI with new values
	updateGUI();
}

void NormDock::setNormMode(representation::t type,MultiImg::NormMode mode)
{
	if(!(representation::IMG == type || representation::GRAD == type ))
		return;
	modes[type] = mode;
	// update GUI with new values
	updateGUI();
}

void NormDock::setNormTarget(representation::t type)
{
	if(!(representation::IMG == type || representation::GRAD == type ))
		return;
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

void NormDock::updateGUI()
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
	normMaxBox->blockSignals(true);
	normMinBox->setEnabled(false);
	normMaxBox->setEnabled(false);
	// FIXME when switching from THEORETICAL or FIXED to OBSERVED,
	// the actual observed values will not be displayed.
	normMinBox->setValue(ranges[normTarget].min);
	normMaxBox->setValue(ranges[normTarget].max);
	if(modes[normTarget] == MultiImg::NORM_FIXED) {
		normMinBox->setEnabled(true);
		normMaxBox->setEnabled(true);
	} else if (modes[normTarget] == MultiImg::NORM_THEORETICAL) {
		// FIXME assuming image depth is 8-bit always.
		normMinBox->setValue(0.);
		normMaxBox->setValue(255.);
	} else { // OBSERVED
		// nothing
	}
	normMinBox->blockSignals(false);
	normMaxBox->blockSignals(false);
}

void NormDock::processNormModeSelected(int idx)
{
	modes[normTarget] = static_cast<MultiImg::NormMode>(idx);
	updateGUI();
}

void NormDock::processMinValueChanged()
{
	//GGDBG_CALL();
	const double val = normMinBox->value();
	ranges[normTarget].min = val;
	//GGDBGM(ranges[normTarget].max << endl);
	ranges[normTarget].max = max(val, ranges[normTarget].max);
	//GGDBGM(ranges[normTarget].max << endl);
	updateGUI();
}

void NormDock::processMaxValueChanged()
{
	//GGDBG_CALL();
	const double val = normMaxBox->value();
	ranges[normTarget].max = val;
	ranges[normTarget].min = min(val, ranges[normTarget].min);
	updateGUI();
}


