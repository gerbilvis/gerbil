#include "ussegmentationdock.h"

// vole modules
#include <meanshift.h>
#include <meanshift_shell.h>

#include "../commandrunner.h"

UsSegmentationDock::UsSegmentationDock(QWidget *parent) :
    QDockWidget(parent)
{
	Ui::UsSegmentationDock::setupUi(this);
	// FIXME nbands
	initUi(31);
}

#ifdef WITH_SEG_MEANSHIFT
void UsSegmentationDock::initUi(size_t nbands)
{
	usMethodBox->addItem("Meanshift", 0);
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

	usBandsSpinBox->setValue(nbands);
	usBandsSpinBox->setMaximum(nbands);

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
	// allow only one runner at a time (UI enforces that)
	assert(usRunner == NULL);
	usRunner = new CommandRunner();

	int method = usMethodBox->itemData(usMethodBox->currentIndex()).value<int>();

	if (findKL) { // run MeanShift::findKL()
		usRunner->cmd = new vole::MeanShiftShell();
		vole::MeanShiftConfig &config = ((vole::MeanShiftShell *) usRunner->cmd)->config;

		config.batch = true;
		config.findKL = true;
		config.k = usPilotKSpinBox->value();
		config.K = usFindKLKmaxBox->value();
		config.L = usFindKLLmaxBox->value();
		config.Kmin = usFindKLKMinBox->value();
		config.Kjump = usFindKLKStepBox->value();
		config.epsilon = usFindKLEpsilonBox->value();
	} else if (method == 0) { // Meanshift
		usRunner->cmd = new vole::MeanShiftShell();
		vole::MeanShiftConfig &config = ((vole::MeanShiftShell *) usRunner->cmd)->config;

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

// medianshift and probshift to be removed from GUI
//#ifdef WITH_SEG_MEDIANSHIFT
//	} else if (method == 1) { // Medianshift
//		usRunner->cmd = new vole::MedianShiftShell();
//		vole::MedianShiftConfig &config = ((vole::MedianShiftShell *) usRunner->cmd)->config;

//		config.K = usKSpinBox->value();
//		config.L = usLSpinBox->value();
//		config.k = usPilotKSpinBox->value();
//		config.skipprop = usSkipPropCheckBox->isChecked();
//#endif
//#ifdef WITH_SEG_PROBSHIFT
//	} else { // Probabilistic Shift
//		usRunner->cmd = new vole::ProbShiftShell();
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
	}

	// connect runner with progress bar, cancel button and finish-slot
	connect(usRunner, SIGNAL(progressChanged(int)), usProgressBar, SLOT(setValue(int)));
	connect(usCancelButton, SIGNAL(clicked()), usRunner, SLOT(terminate()));

	qRegisterMetaType< std::map<std::string, boost::any> >("std::map<std::string, boost::any>");
	connect(usRunner, SIGNAL(success(std::map<std::string,boost::any>)), this, SLOT(segmentationApply(std::map<std::string,boost::any>)));
	connect(usRunner, SIGNAL(finished()), this, SLOT(segmentationFinished()));

	usProgressWidget->show();
	usSettingsWidget->setDisabled(true);

	// prepare input image
	boost::shared_ptr<multi_img> input;
	{
/*TODO		SharedMultiImgBaseGuard guard(*image_lim);
		assert(0 != &**image_lim);
		// FIXME 2013-04-11 georg altmann:
		// not sure what this code is really doing, but this looks like a problem:
		// is input sharing image data with image_lim?
		// If so, another thread could overwrite data while image segmentation is working on it,
		// since there is no locking (unless multi_img does implicit copy on write?).
		input = boost::shared_ptr<multi_img>(
					new multi_img(**image_lim, roi)); // image data is not copied
*/
	}
	int numbands = usBandsSpinBox->value();
	bool gradient = usGradientCheckBox->isChecked();

	if (numbands > 0 && numbands < (int) input->size()) {
		boost::shared_ptr<multi_img> input_tmp(new multi_img(input->spec_rescale(numbands)));
		input = input_tmp;
	}

	if (gradient) {
		// copy needed here
		multi_img loginput(*input);
		loginput.apply_logarithm();
		input = boost::shared_ptr<multi_img>(new multi_img(loginput.spec_gradient()));
	}

	usRunner->input["multi_img"] = input;

	usRunner->start();
}

void UsSegmentationDock::segmentationFinished() {
	if (usRunner->abort) {
		// restore Cancel button
		usCancelButton->setEnabled(true);
		usCancelButton->setText("Cancel");
	}

	// hide progress, re-enable settings
	usProgressWidget->hide();
	usSettingsWidget->setEnabled(true);

	/// clean up runner
	delete usRunner;
	usRunner = NULL;
}

void UsSegmentationDock::segmentationApply(std::map<std::string, boost::any> output) {
	if (output.count("labels")) {
		boost::shared_ptr<cv::Mat1s> labelMask = boost::any_cast< boost::shared_ptr<cv::Mat1s> >(output["labels"]);
		// TODO: assert size?, emit signal for lm
		// TODO setLabels(*labelMask);
	}

	if (output.count("findKL.K") && output.count("findKL.L")) {
		int foundK = boost::any_cast<int>(output["findKL.K"]);
		int foundL = boost::any_cast<int>(output["findKL.L"]);
		usFoundKLLabel->setText(QString("Found values: K=%1 L=%2").arg(foundK).arg(foundL));
		usFoundKLWidget->show();
	}
}
#else // method stubs as using define in header does not work (moc problem?)
// TODO
// 1. ifdef on seg dock header
// 2. probshift, medianshift raus. (auswahl dropbox lassen)
// 3.
void UsSegmentationDock::startUnsupervisedSeg(bool findKL) {}
void UsSegmentationDock::startFindKL() {}
void UsSegmentationDock::segmentationFinished() {}
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
#endif

