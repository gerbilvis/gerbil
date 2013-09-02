#include "ussegmentationdock.h"

#include "../commandrunner.h"
#include "../gerbil_gui_debug.h"

// vole modules
#include <meanshift.h>
#include <meanshift_shell.h>

UsSegmentationDock::UsSegmentationDock(QWidget *parent) :
	QDockWidget(parent),
	nBandsOld(-1)
{
	Ui::UsSegmentationDock::setupUi(this);

	initUi();
}

#ifdef WITH_SEG_MEANSHIFT
void UsSegmentationDock::initUi()
{
	usMethodBox->addItem("FAMS", 0);
	usMethodBox->addItem("PSPMS", 3);
	usMethodBox->addItem("FSPMS", 4);

//#ifdef WITH_SEG_MEDIANSHIFT
//	usMethodBox->addItem("Medianshift", 1);
//#endif
//#ifdef WITH_SEG_PROBSHIFT
//	usMethodBox->addItem("Probabilistic Shift", 2);
//#endif
	usMethodChanged(0); // set default state

	usInitMethodBox->addItem("all", vole::ALL);
	usInitMethodBox->addItem("jump", vole::JUMP);
	usInitMethodBox->addItem("percent", vole::PERCENT);
	usInitMethodChanged(0);

	usBandwidthBox->addItem("adaptive");
	usBandwidthBox->addItem("fixed");
	usBandwidthMethodChanged("adaptive");

	// will be set later by controller
	usBandsSpinBox->setValue(-1);
	usBandsSpinBox->setMaximum(-1);

	// we do not expose parameters that nobody uses, incl. findKL functionality
	usLshWidget->hide();
	usBandwidthWidget->hide();
	findKLWidget->hide();
	usSkipPropWidget->hide();
	usMSPPWidget->hide();
	usSpectralWidget->hide();

	// we do not expose the density estimation functionality
	usInitWidget->hide();
	// we also do not expose options exclusive to unavailable methods
//#ifndef WITH_SEG_MEDIANSHIFT
//	usSkipPropWidget->hide();
//#endif
//#ifndef WITH_SEG_PROBSHIFT
//	usSpectralWidget->hide();
//	usMSPPWidget->hide();
//#endif

	usInitJumpWidget->hide();
	usInitPercentWidget->hide();
	usFoundKLWidget->hide();
	usProgressWidget->hide();

	connect(usGoButton, SIGNAL(clicked()),
			this, SLOT(startUnsupervisedSeg()));
	connect(usFindKLGoButton, SIGNAL(clicked()),
			this, SLOT(startFindKL()));
	connect(usCancelButton, SIGNAL(clicked()),
			this, SLOT(unsupervisedSegCancelled()));

	connect(usMethodBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(usMethodChanged(int)));

	connect(usLshCheckBox, SIGNAL(toggled(bool)),
			usLshWidget, SLOT(setEnabled(bool)));

	connect(usBandwidthBox, SIGNAL(currentIndexChanged(const QString&)),
			this, SLOT(usBandwidthMethodChanged(const QString&)));

	connect(usInitMethodBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(usInitMethodChanged(int)));

	connect(usSpectralCheckBox, SIGNAL(toggled(bool)),
			usSpectralConvCheckBox, SLOT(setEnabled(bool)));
	connect(usSpectralCheckBox, SIGNAL(toggled(bool)),
			usSpectralMinMaxWidget, SLOT(setEnabled(bool)));

	/// pull default values from temporary instance of config class
	vole::MeanShiftConfig def;
	usKSpinBox->setValue(def.K);
	usLSpinBox->setValue(def.L);
	/// TODO: random seed box
	usPilotKSpinBox->setValue(def.k);
	usInitMethodBox->setCurrentIndex(
			usInitMethodBox->findData(def.starting));
	usInitJumpBox->setValue(def.jump);
	usFixedBWSpinBox->setValue(def.bandwidth);
	usFindKLKMinBox->setValue(def.Kmin);
	usFindKLKStepBox->setValue(def.Kjump);
	usFindKLEpsilonBox->setValue(def.epsilon);

//#ifdef WITH_SEG_PROBSHIFT
//	vole::ProbShiftConfig def_ps;
//	usProbShiftMSPPAlphaSpinBox->setValue(def_ps.msBwFactor);
//#endif
}

void UsSegmentationDock::usBandwidthMethodChanged(const QString &current) {
	if (current == "fixed") {
		usAdaptiveBWWidget->hide();
		usFixedBWWidget->show();
	} else if (current == "adaptive") {
		usFixedBWWidget->hide();
		usAdaptiveBWWidget->show();
	} else {
		assert(0);
	}
}

void UsSegmentationDock::unsupervisedSegCancelled() {
	usCancelButton->setDisabled(true);
	usCancelButton->setText("Please wait...");
	/// runner->terminate() will be called by the Cancel button
}

void UsSegmentationDock::startFindKL()
{
	startUnsupervisedSeg(true);
}

void UsSegmentationDock::startUnsupervisedSeg(bool findKL)
{
	int method = usMethodBox->itemData(usMethodBox->currentIndex()).value<int>();
	vole::Command *cmd;
	if (findKL) { // run MeanShift::findKL()
		cmd = new vole::MeanShiftShell();
		vole::MeanShiftConfig &config =
				static_cast<vole::MeanShiftShell*>(cmd)->config;

		config.batch = true;
		config.findKL = true;
		config.k = usPilotKSpinBox->value();
		config.K = usFindKLKmaxBox->value();
		config.L = usFindKLLmaxBox->value();
		config.Kmin = usFindKLKMinBox->value();
		config.Kjump = usFindKLKStepBox->value();
		config.epsilon = usFindKLEpsilonBox->value();
	} else if (method == 0 || method == 3) { // Meanshift
		cmd = new vole::MeanShiftShell();
		vole::MeanShiftConfig &config =
				static_cast<vole::MeanShiftShell*>(cmd)->config;

		// fixed settings
		config.batch = true;
		if (method == 3) {
			/* if combination of gradient and PSPMS requested, we assume that
			   the user wants our best-working method in paper (sp_withGrad)
			   TODO: most of this is better done in model, as it is internals
			 */
			config.sp_withGrad = usGradientCheckBox->isChecked();

			config.starting = vole::SUPERPIXEL;

			config.superpixel.eqhist=1;
			config.superpixel.c=0.05;
			config.superpixel.min_size=5;
			config.superpixel.similarity.measure=vole::SPEC_INF_DIV;
		}

		config.use_LSH = usLshCheckBox->isChecked();
		//config.K = usKSpinBox->value();
		//config.L = usLSpinBox->value();

		//config.starting = (vole::ms_sampling) usInitMethodBox->itemData(usInitMethodBox->currentIndex()).value<int>();
		//config.percent = usInitPercentBox->value();
		//config.jump = usInitJumpBox->value();
		//config.k = usPilotKSpinBox->value();

		//if (usBandwidthBox->currentText() == "fixed") {
		//	config.bandwidth = usFixedBWSpinBox->value();
		//} else {
		//	config.bandwidth = 0;
		//}

// old: medianshift and probshift removed from GUI
//#ifdef WITH_SEG_MEDIANSHIFT
//	} else if (method == 1) { // Medianshift
//		cmd = new vole::MedianShiftShell();
//		vole::MedianShiftConfig &config = ((vole::MedianShiftShell *) usRunner->cmd)->config;

//		config.K = usKSpinBox->value();
//		config.L = usLSpinBox->value();
//		config.k = usPilotKSpinBox->value();
//		config.skipprop = usSkipPropCheckBox->isChecked();
//#endif
//#ifdef WITH_SEG_PROBSHIFT
//	} else { // Probabilistic Shift
//		cmd = new vole::ProbShiftShell();
//		vole::ProbShiftConfig &config = ((vole::ProbShiftShell *) usRunner->cmd)->config;

//		config.useLSH = usLshCheckBox->isChecked();
//		config.lshK = usKSpinBox->value();
//		config.lshL = usLSpinBox->value();

//		config.useSpectral = usSpectralCheckBox->isChecked();
//		config.useConverged = usSpectralConvCheckBox->isChecked();
//		config.minClusts = usSpectralMinBox->value();
//		config.maxClusts = usSpectralMaxBox->value();
//		config.useMeanShift = usProbShiftMSPPCheckBox->isChecked();
//		config.msBwFactor = usProbShiftMSPPAlphaSpinBox->value();
//#endif
	} else if (method == 4) { // FSPMS
		cmd = new vole::MeanShiftShell();
		vole::MeanShiftConfig &config =
				static_cast<vole::MeanShiftShell*>(cmd)->config;

		// fixed settings
		config.batch = true;

		config.use_LSH = usLshCheckBox->isChecked();
		config.K = usKSpinBox->value();
		config.L = usLSpinBox->value();

		config.starting = (vole::ms_sampling) usInitMethodBox->itemData(usInitMethodBox->currentIndex()).value<int>();
		config.percent = usInitPercentBox->value();
		config.jump = usInitJumpBox->value();
		config.k = usPilotKSpinBox->value();

		if (usBandwidthBox->currentText() == "fixed") {
			config.bandwidth = usFixedBWSpinBox->value();
		} else {
			config.bandwidth = 0;
		}
	}

	connect(usCancelButton, SIGNAL(clicked()),
			this, SLOT(cancel()));

	usProgressWidget->show();
	usSettingsWidget->setDisabled(true);

	int numbands = usBandsSpinBox->value();
	bool gradient = usGradientCheckBox->isChecked();

	emit segmentationRequested(cmd, numbands, gradient);
}
#else // method stubs as using define in header does not work (moc problem?)
void UsSegmentationDock::startUnsupervisedSeg(bool findKL) {}
void UsSegmentationDock::startFindKL() {}
void UsSegmentationDock::segmentationApply(std::map<std::string, boost::any>) {}
void UsSegmentationDock::usMethodChanged(int idx) {}
void UsSegmentationDock::usInitMethodChanged(int idx) {}
void UsSegmentationDock::usBandwidthMethodChanged(const QString &current) {}
void UsSegmentationDock::unsupervisedSegCancelled() {}
#endif // WITH_SEG_MEANSHIFT



#ifdef WITH_SEG_MEANSHIFT
void UsSegmentationDock::usMethodChanged(int idx)
{
	// idx: 0 Meanshift, 1 Medianshift, 2 Probabilistic Shift
	usSkipPropWidget->setEnabled(idx == 1);
	usSpectralWidget->setEnabled(idx == 2);
	usMSPPWidget->setEnabled(idx == 2);
}

void UsSegmentationDock::usInitMethodChanged(int idx)
{
	switch (usInitMethodBox->itemData(idx).toInt()) {
	case vole::JUMP:
		usInitPercentWidget->hide();
		usInitJumpWidget->show();
		break;
	case vole::PERCENT:
		usInitJumpWidget->hide();
		usInitPercentWidget->show();
		break;
	default:
		usInitJumpWidget->hide();
		usInitPercentWidget->hide();
	}
}
int UsSegmentationDock::updateProgress(int percent)
{
	usProgressBar->setValue(percent);
}

int UsSegmentationDock::processResultKL(int k, int l)
{
	usFoundKLLabel->setText(QString("Found values: K=%1 L=%2").arg(k).arg(l));
	usFoundKLWidget->show();
}

void UsSegmentationDock::processSegmentationCompleted()
{
	// hide progress, re-enable settings
	usProgressWidget->hide();
	usSettingsWidget->setEnabled(true);

}

void UsSegmentationDock::cancel()
{
	emit cancelSegmentationRequested();

	// restore Cancel button
	usCancelButton->setEnabled(true);
	usCancelButton->setText("Cancel");


	// hide progress, re-enable settings
	usProgressWidget->hide();
	usSettingsWidget->setEnabled(true);
}

void UsSegmentationDock::setNumBands(int nBands)
{
	int nBandsSpin = usBandsSpinBox->value();
	//GGDBGM(nBandsSpin <<" " << nBands<<" " <<nBandsOld<<endl);

	usBandsSpinBox->setMaximum(nBands);

	// first time init
	if(nBandsSpin == -1) {
		usBandsSpinBox->setMinimum(0);
		usBandsSpinBox->setValue(nBands);
		nBandsOld = nBands;
	} else if(nBandsSpin == nBandsOld) {
		// Track the global number of bands until the user manually changes the
		// value in the spinbox.
		usBandsSpinBox->setValue(nBands);
		nBandsOld = nBands;
	}
}

#endif

