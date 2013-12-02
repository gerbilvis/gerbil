#include "banddock.h"
#include "../widgets/bandview.h"
#include "../widgets/graphsegwidget.h"
#include "../iogui.h"

#include "../gerbil_gui_debug.h"

/** Return a 32x32px icon filled with color. */
static QIcon colorIcon(const QColor &color)
{
	QPixmap pm(32, 32);
	pm.fill(color);
	return QIcon(pm);
}

BandDock::BandDock(cv::Rect fullImgSize, QWidget *parent)
	: QDockWidget(parent), fullImgSize(fullImgSize),
	  curRepr(representation::IMG), curBandId(0)
{
	setupUi(this);
	initUi();
}

BandDock::~BandDock()
{
}

void BandDock::initUi()
{
	// initialize band view
	view->installEventFilter(this); // needed for enter/leave events
	view->init();
	bv = new BandView();
	view->setScene(bv);
	connect(bv, SIGNAL(newContentRect(QRect)),
			view, SLOT(fitContentRect(QRect)));

	// add graphseg control widget
	gs = new GraphSegWidget(view);
	bv->offBottom = AutohideWidget::OutOffset;
	view->addWidget(AutohideWidget::BOTTOM, gs);

	connect(bv, SIGNAL(newSizeHint(QSize)),
			view, SLOT(updateSizeHint(QSize)));

	connect(gs, SIGNAL(requestLoadSeeds()),
			this, SLOT(loadSeeds()));
	connect(gs, SIGNAL(requestClearSeeds()),
			bv, SLOT(clearSeeds()));

	connect(markerSelector, SIGNAL(currentIndexChanged(int)),
			this, SLOT(processMarkerSelectorIndexChanged(int)));

	connect(alphaSlider, SIGNAL(valueChanged(int)),
			bv, SLOT(applyLabelAlpha(int)));

	connect(clearButton, SIGNAL(clicked()),
			this, SLOT(clearLabel()));
	connect(bv, SIGNAL(clearRequested()),
			this, SLOT(clearLabel()));

	/* when applybutton is pressed, bandView commits full label matrix */
	connect(applyButton, SIGNAL(clicked()),
			bv, SLOT(commitLabels()));
	// The apply button is unecessary now, since
	// label update is triggered by timer. Hide it for now.
	applyButton->setVisible(false);

	connect(gs, SIGNAL(requestToggleSeedMode(bool)),
			this, SLOT(graphSegModeToggled(bool)));
	connect(gs, SIGNAL(requestToggleSeedMode(bool)),
			bv, SLOT(toggleSeedMode(bool)));

	connect(this, SIGNAL(currentLabelChanged(int)),
			bv, SLOT(setCurrentLabel(int)));

	bv->initUi();
}

void BandDock::changeBand(representation::t repr, int bandId,
						  QPixmap band, QString desc)
{
	curRepr = repr;
	curBandId = bandId;

//TODO	bv->setEnabled(true);
	bv->setPixmap(band);
	setWindowTitle(desc);
}

void BandDock::processImageUpdate(representation::t type)
{
	if (type == curRepr) {
		// our view became invalid. fetch new one.
		emit bandRequested(curRepr, curBandId);
	}
}

void BandDock::clearLabel()
{
	// FIXME need to stop label timer of bandview
	emit clearLabelRequested(bv->getCurrentLabel());
}

void BandDock::processMarkerSelectorIndexChanged(int index)
{
	if (index < 0)	// empty selection, during initialization
		return;

	index += 1; // we start with 1, combobox with 0

	int nlabels = labelColors.count();

	if (nlabels && index == nlabels) {	// new label requested

		// commit uncommitted label changes in the bandview
		bv->commitLabelChanges();
		// issue creation of a new label
		emit newLabelRequested();

		// select that label, will return back here into the else() case
		markerSelector->setCurrentIndex(index-1);
	} else {
		// propagate label change
		emit currentLabelChanged(index);
	}
}

bool BandDock::eventFilter(QObject *obj, QEvent *event)
{
	if (event->type() == QEvent::Enter)
		bv->enterEvent();
	if (event->type() == QEvent::Leave)
		bv->leaveEvent();

	// continue with standard event processing
	return QObject::eventFilter(obj, event);
}

void BandDock::processLabelingChange(const cv::Mat1s &labels,
									   const QVector<QColor> &colors,
									   bool colorsChanged)
{
	if (!colors.empty()) {
		// store a local copy of the color array
		labelColors = colors;
		/* use colors for our awesome label menu (rebuild everything) */
		// block signals to not fire spurious label changed events
		markerSelector->blockSignals(true);
		markerSelector->clear();
		for (int i = 1; i < colors.size(); ++i) // 0 is index for unlabeled
		{
			markerSelector->addItem(colorIcon(colors.at(i)), "");
		}
		markerSelector->addItem(QIcon::fromTheme("list-add"), "");
		markerSelector->setCurrentIndex(bv->getCurrentLabel() - 1);
		markerSelector->blockSignals(false);

		/* make sure our current label fits -> this does not only affect bv! */
		int oldindex = bv->getCurrentLabel();
		if (oldindex < 1 || oldindex >= labelColors.count()) {
			// set to the always valid default (propagates to bv)
			markerSelector->setCurrentIndex(0);
		}
	}

	// tell bandview about the update as well
	bv->updateLabeling(labels, colors, colorsChanged);
}

void BandDock::processLabelingChange(const cv::Mat1s &labels,
									   const cv::Mat1b &mask)
{
	bv->updateLabeling(labels, mask);
}

void BandDock::graphSegModeToggled(bool)
{
}

void BandDock::loadSeeds()
{
	IOGui io("Seed Image File", "seed image", this);
	cv::Mat1s seeding = io.readFile(QString(), 0,
									fullImgSize.height, fullImgSize.width);
	if (seeding.empty())
		return;

	bv->setSeedMap(seeding);
}


