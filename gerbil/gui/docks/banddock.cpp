#include "banddock.h"
#include "../iogui.h"
#include "ui_banddock.h"

#include "../gerbil_gui_debug.h"

/** Return a 32x32px icon filled with color. */
static QIcon colorIcon(const QColor &color)
{
	QPixmap pm(32, 32);
	pm.fill(color);
	return QIcon(pm);
}

BandDock::BandDock(cv::Rect fullImgSize, QWidget *parent)
	: QDockWidget(parent), fullImgSize(fullImgSize)
{
	setupUi(this);
	initUi();
}

BandDock::~BandDock()
{
}

void BandDock::initUi()
{
	connect(gs, SIGNAL(requestLoadSeeds()),
			this, SLOT(loadSeeds()));

	connect(markerSelector, SIGNAL(currentIndexChanged(int)),
			bv, SLOT(changeCurrentLabel(int)));
	connect(markerSelector, SIGNAL(currentIndexChanged(int)),
			this, SLOT(processMarkerSelectorIndexChanged(int)));

	connect(alphaSlider, SIGNAL(valueChanged(int)),
			bv, SLOT(applyLabelAlpha(int)));

	connect(clearButton, SIGNAL(clicked()),
			this, SLOT(clearLabelOrSeeds()));

	// -> DockController
	//	connect(bandView, SIGNAL(alteredLabels(const cv::Mat1s&, const cv::Mat1b&)),
	//			this, SIGNAL(alterLabelingRequested(cv::Mat1s,cv::Mat1b)));
	//	connect(bandView, SIGNAL(newLabeling(const cv::Mat1s&)),
	//			this, SIGNAL(newLabelingRequested(cv::Mat1s)));

	/* when applybutton is pressed, bandView commits full label matrix */
	connect(applyButton, SIGNAL(clicked()),
			bv, SLOT(commitLabels()));
	// The apply button is unecessary now, since
	// label update is triggered by timer. Hide it for now.
	applyButton->setVisible(false);

	connect(graphsegButton, SIGNAL(toggled(bool)),
			this, SLOT(graphSegModeToggled(bool)));
	connect(graphsegButton, SIGNAL(toggled(bool)),
			bv, SLOT(toggleSeedMode(bool)));

	bv->initUi();
	gs->setVisible(false);
}

void BandDock::changeBand(representation::t repr, int bandId,
						  QPixmap band, QString desc)
{
	curRepr = repr;
	curBandId = bandId;

	bv->setEnabled(true);
	bv->setPixmap(band);
	setWindowTitle(desc);
}

void BandDock::processSeedingDone()
{
	graphsegButton->setChecked(false);
}

void BandDock::clearLabelOrSeeds()
{
	// FIXME need to stop label timer of bandview

	if (bv->isSeedModeEnabled()) {
		bv->clearSeeds();
	} else {
		emit clearLabelRequested(bv->getCurLabel());
	}
}

void BandDock::processMarkerSelectorIndexChanged(int idx)
{
	// notify other parties
	emit currentLabelChanged(idx);

	///GGDBGM(format("idx=%1%")%idx<<endl);
	if (idx < 0)	// empty selection, during initialization
		return;
	idx += 1; // we start with 1, combobox with 0

	int nlabels = labelColors.count();

	if (nlabels && idx == nlabels) {
		// new label requested

		// commit uncommitted label changes in the bandview
		bv->commitLabelChanges();
		emit newLabelRequested();

		// will not loop.
		markerSelector->setCurrentIndex(idx-1);
	} else {
		bv->changeCurrentLabel(idx);
	}
}

void BandDock::processLabelingChange(const cv::Mat1s &labels,
									   const QVector<QColor> &colors,
									   bool colorsChanged)
{
	if (!colors.empty()) {
		//GGDBGM("colors.size()=" << colors.size() << endl);
		// store a local copy of the color array
		labelColors = colors;
		// use colors for our awesome label menu (rebuild everything)
		markerSelector->clear();
		for (int i = 1; i < colors.size(); ++i) // 0 is index for unlabeled
		{
			markerSelector->addItem(colorIcon(colors.at(i)), "");
		}
		markerSelector->addItem(QIcon(":/toolbar/add"), "");
	}

	// tell bandview about the update as well
	bv->updateLabeling(labels, colors, colorsChanged);
}

void BandDock::processLabelingChange(const cv::Mat1s &labels,
									   const cv::Mat1b &mask)
{
	//GGDBG_CALL();
	bv->updateLabeling(labels, mask);
}

void BandDock::graphSegModeToggled(bool enable)
{
	gs->setVisible(enable);
}

void BandDock::loadSeeds()
{
	IOGui io("Seed Image File", "seed image", this);
	cv::Mat1s seeding = io.readFile(QString(), 0,
									fullImgSize.height, fullImgSize.width);
	if (seeding.empty())
		return;

	bv->setSeedMap(seeding);

	// now make sure we are in seed mode
	if (!graphsegButton->isChecked()) {
		graphsegButton->toggle();
	}
}


