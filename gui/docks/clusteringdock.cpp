#include "clusteringdock.h"

#include "../commandrunner.h"
#include "../gerbil_gui_debug.h"
#include <model/clustering/clusteringmethod.h>

// vole modules
#include <meanshift.h>
#include <meanshift_shell.h>
#include <meanshift_sp.h>

ClusteringDock::ClusteringDock(QWidget *parent) :
	QDockWidget(parent)
{
	Ui::ClusteringDock::setupUi(this);

	initUi();
}

#ifdef WITH_SEG_MEANSHIFT
void ClusteringDock::initUi()
{
	usMethodBox->addItem("Accurate (FAMS)", ClusteringMethod::FAMS);
	usMethodBox->addItem("Fast (PSPMS)",    ClusteringMethod::PSPMS);
	usMethodBox->addItem("Fastest (FSPMS)", ClusteringMethod::FSPMS);

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
	usCancelButton->setText("Please wait...");
	/// runner->terminate() will be called by the Cancel button
}

void ClusteringDock::startUnsupervisedSeg()
{
	using namespace seg_meanshift;

	ClusteringMethod::t method =
			usMethodBox->itemData(usMethodBox->currentIndex())
			.value<ClusteringMethod::t>();
	// Command will be deleted by CommandRunner on destruction.
	shell::Command *cmd;
	const bool onGradient = usGradientCheckBox->isChecked();

	// Meanshift
	if (ClusteringMethod::FAMS == method || ClusteringMethod::PSPMS == method) {
		cmd = new MeanShiftShell();
		MeanShiftConfig &config = static_cast<MeanShiftShell*>(cmd)->config;

		// fixed settings
		if (ClusteringMethod::PSPMS == method) {
			/* if combination of gradient and PSPMS requested, we assume that
			   the user wants our best-working method in paper (sp_withGrad)
			   TODO: most of this is better done in model, as it is internals
			 */
			config.sp_withGrad = onGradient;

			config.starting = SUPERPIXEL;

			config.superpixel.eqhist=1;
			config.superpixel.c=0.05f;
			config.superpixel.min_size=5;
			config.superpixel.similarity.function
					= similarity_measures::SPEC_INF_DIV;
		}

		config.use_LSH = usLshCheckBox->isChecked();
	} else if (ClusteringMethod::FSPMS == method) { // FSPMS
		cmd = new MeanShiftSP();
		MeanShiftConfig &config = static_cast<MeanShiftSP*>(cmd)->config;

		// fixed settings
		/* see method == ClusteringMethod::PSPMS
		 */
		config.sp_withGrad = onGradient;
		config.superpixel.eqhist=1;
		config.superpixel.c=0.05f;
		config.superpixel.min_size=5;
		config.superpixel.similarity.function
				= similarity_measures::SPEC_INF_DIV;
		config.sp_weight = 2;

		config.use_LSH = usLshCheckBox->isChecked();
	}

	connect(usCancelButton, SIGNAL(clicked()),
			this, SLOT(cancel()));

	usProgressWidget->show();
	usSettingsWidget->setDisabled(true);

	emit segmentationRequested(cmd, onGradient);
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
#endif

