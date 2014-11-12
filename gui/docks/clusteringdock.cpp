#include "clusteringdock.h"

#include "../gerbil_gui_debug.h"

ClusteringDock::ClusteringDock(QWidget *parent) :
	QDockWidget(parent)
{
	Ui::ClusteringDock::setupUi(this);

	initUi();
}

#ifdef WITH_SEG_MEANSHIFT

// because writing QVariant::fromValue(xxx) is just too much boilerplate
static inline QVariant qv(ClusteringMethod::t cm) {
	return QVariant::fromValue(cm);
}

void ClusteringDock::initUi()
{
	usMethodBox->addItem("Accurate (FAMS)", qv(ClusteringMethod::FAMS));
	usMethodBox->addItem("Fast (PSPMS)",    qv(ClusteringMethod::PSPMS));
	usMethodBox->addItem("Fastest (FSPMS)", qv(ClusteringMethod::FSPMS));

	// don't show progress widget at startup
	usProgressWidget->hide();

	connect(usGoButton, SIGNAL(clicked()),
			this, SLOT(startUnsupervisedSeg()));
	connect(usCancelButton, SIGNAL(clicked()),
			this, SLOT(unsupervisedSegCancelled()));
	connect(usCancelButton, SIGNAL(clicked()),
			this, SLOT(cancel()));
	connect(usMethodBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(usMethodChanged(int)));
	connect(usLshCheckBox, SIGNAL(toggled(bool)),
			usLshWidget, SLOT(setEnabled(bool)));

	// set default state to PSPMS, after connecting signals
	usMethodBox->setCurrentIndex(ClusteringMethod::PSPMS); //
}

void ClusteringDock::unsupervisedSegCancelled() {
	usCancelButton->setDisabled(true);
	usCancelButton->setText("Please wait...");
	/// runner->terminate() will be called by the Cancel button
}

void ClusteringDock::startUnsupervisedSeg()
{
	ClusteringMethod::t method =
			usMethodBox->itemData(usMethodBox->currentIndex())
			.value<ClusteringMethod::t>();

	representation::t repr;
	if (usGradientCheckBox->isChecked()) {
		repr = representation::GRAD;
	} else {
		repr = representation::NORM;
	}

	bool lsh = usLshCheckBox->isChecked();

	usProgressWidget->show();
	usSettingsWidget->setDisabled(true);
	emit segmentationRequested(ClusteringRequest(method, repr, lsh));
}
#else // method stubs as using define in header does not work (moc problem?)
void ClusteringDock::startUnsupervisedSeg() {}
void ClusteringDock::usMethodChanged(int idx) {}
void ClusteringDock::unsupervisedSegCancelled() {}
#endif // WITH_SEG_MEANSHIFT

#ifdef WITH_SEG_MEANSHIFT
void ClusteringDock::usMethodChanged(int idx)
{
	// use to change available (method-dependent) options
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
#endif // WITH_SEG_MEANSHIFT

