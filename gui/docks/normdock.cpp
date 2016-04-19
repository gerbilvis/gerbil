#include "normdock.h"
#include "ui_normdock.h"

#include "../gerbil_gui_debug.h"

#include <algorithm>

// for DEBUG
#ifdef GGDBG_MODULE
static std::ostream &operator<<(std::ostream& os, const multi_img::Range& r)
{
	os << boost::format("[%1%,%2%]") % r.min % r.max;
	return os;
}
#endif // GGDBG

NormDock::NormDock(QWidget *parent) :
	QDockWidget(parent)
{
	// init with zero ranges
	ranges.insert(representation::IMG, multi_img::Range());
	ranges.insert(representation::GRAD, multi_img::Range());

	modes.insert(representation::IMG, multi_img::NORM_OBSERVED);
	modes.insert(representation::GRAD, multi_img::NORM_OBSERVED);

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

void NormDock::setNormRange(representation::t type, const multi_img::Range& range)
{
	//GGDBGM(type << " " << range << endl);
	if(!(representation::IMG == type || representation::GRAD == type ))
		return;
	ranges[type] = range;
	// update GUI with new values
	updateGUI();
}

void NormDock::setNormMode(representation::t type,multi_img::NormMode mode)
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
	if(modes[normTarget] == multi_img::NORM_FIXED) {
		normMinBox->setEnabled(true);
		normMaxBox->setEnabled(true);
	} else if (modes[normTarget] == multi_img::NORM_THEORETICAL) {
		// FIXME assuming image depth is 8-bit always.
		// HACK
		if(representation::GRAD == normTarget) {
			normMinBox->setValue(-5.54);
			normMaxBox->setValue(5.54);
		} else { // IMG
			normMinBox->setValue(0.);
			normMaxBox->setValue(255.);
		}
	} else { // OBSERVED
		// nothing
	}
	normMinBox->blockSignals(false);
	normMaxBox->blockSignals(false);
}

void NormDock::processNormModeSelected(int idx)
{
	modes[normTarget] = static_cast<multi_img::NormMode>(idx);
	updateGUI();
}

void NormDock::processMinValueChanged()
{
	//GGDBG_CALL();
	const double val = normMinBox->value();
	ranges[normTarget].min = val;
	//GGDBGM(ranges[normTarget].max << endl);
	ranges[normTarget].max = std::max(val, (double)ranges[normTarget].max);
	//GGDBGM(ranges[normTarget].max << endl);
	updateGUI();
}

void NormDock::processMaxValueChanged()
{
	//GGDBG_CALL();
	const double val = normMaxBox->value();
	ranges[normTarget].max = val;
	ranges[normTarget].min = std::min(val, (double)ranges[normTarget].min);
	updateGUI();
}


