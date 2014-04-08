#include "clusteringdock.h"

#include "../commandrunner.h"
#include "../gerbil_gui_debug.h"

// vole modules
#include <meanshift.h>
#include <meanshift_shell.h>
#include <meanshift_sp.h>

ClusteringDock::ClusteringDock(QWidget *parent) :
	QDockWidget(parent),
	nBandsOld(-1)
{
	Ui::ClusteringDock::setupUi(this);

	initUi();
}

#ifdef WITH_SEG_MEANSHIFT
void ClusteringDock::initUi()
{
	usMethodBox->addItem("Accurate (FAMS)", 0);
	usMethodBox->addItem("Fast (PSPMS)", 3);
	usMethodBox->addItem("Fastest (FSPMS)", 4);

	// will be set later by controller
	usBandsSpinBox->setValue(-1);
	usBandsSpinBox->setMaximum(-1);

	// don't show progress widget at startup
	usProgressWidget->hide();

	connect(usGoButton, SIGNAL(clicked()),
			this, SLOT(startUnsupervisedSeg()));
	connect(usCancelButton, SIGNAL(clicked()),
			this, SLOT(unsupervisedSegCancelled()));

	connect(usMethodBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(usMethodChanged(int)));

	connect(usLshCheckBox, SIGNAL(toggled(bool)),
			usLshWidget, SLOT(setEnabled(bool)));

	// do this after connecting signals
	usMethodBox->setCurrentIndex(1); // set default state to PSPMS
}

void ClusteringDock::unsupervisedSegCancelled() {
	usCancelButton->setDisabled(true);
	usCancelButton->setText("Please wait…");
	/// runner->terminate() will be called by the Cancel button
}

void ClusteringDock::startUnsupervisedSeg()
{
	int method = usMethodBox->itemData(usMethodBox->currentIndex()).value<int>();
	vole::Command *cmd;
	if (method == 0 || method == 3) { // Meanshift
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
			config.superpixel.c=0.05f;
			config.superpixel.min_size=5;
			config.superpixel.similarity.measure=vole::SPEC_INF_DIV;
		}

		config.use_LSH = usLshCheckBox->isChecked();
	} else if (method == 4) { // FSPMS
		cmd = new vole::MeanShiftSP();
		vole::MeanShiftConfig &config =
				static_cast<vole::MeanShiftSP*>(cmd)->config;

		// fixed settings
		config.batch = true;
		/* see method == 3
		 */
		config.sp_withGrad = usGradientCheckBox->isChecked();
		config.superpixel.eqhist=1;
		config.superpixel.c=0.05f;
		config.superpixel.min_size=5;
		config.superpixel.similarity.measure=vole::SPEC_INF_DIV;
		config.sp_weight = 2;

		config.use_LSH = usLshCheckBox->isChecked();
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
void ClusteringDock::startUnsupervisedSeg() {}
void ClusteringDock::segmentationApply(std::map<std::string, boost::any>) {}
void ClusteringDock::usMethodChanged(int idx) {}
void ClusteringDock::unsupervisedSegCancelled() {}
#endif // WITH_SEG_MEANSHIFT



#ifdef WITH_SEG_MEANSHIFT
void ClusteringDock::usMethodChanged(int idx)
{
	// use to change available options
}

void ClusteringDock::updateProgress(int percent)
{
	usProgressBar->setValue(percent);
}

void ClusteringDock::processSegmentationCompleted()
{
	// hide progress, re-enable settings
	usProgressWidget->hide();
	usSettingsWidget->setEnabled(true);

}

void ClusteringDock::cancel()
{
	emit cancelSegmentationRequested();

	// restore Cancel button
	usCancelButton->setEnabled(true);
	usCancelButton->setText("Cancel");


	// hide progress, re-enable settings
	usProgressWidget->hide();
	usSettingsWidget->setEnabled(true);
}

void ClusteringDock::setNumBands(int nBands)
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

