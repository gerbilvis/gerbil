#include "distviewgui.h"
#include "widgets/autohidewidget.h"

//#define GGDBG_MODULE
#include <gerbil_gui_debug.h>

DistViewGUI::DistViewGUI(representation::t type)
	: type(type)
{	
	// setup frame and its UI
	frame = new QWidget();
	frame->setSizePolicy(QSizePolicy::Preferred, // hor
						 QSizePolicy::Expanding); // ver
	ui = new Ui::DistViewGUI();
	ui->setupUi(frame);

	// create viewport
	initVP();

	// create controller widget that will reside inside viewport
	initVC(type);

	// connect and initialize topBar
	initTop();
}

void DistViewGUI::initVP()
{
	// create target widget
	QGLWidget *target = ui->gv->init();

	// create viewport. The viewport is a GraphicsScene
	vp = new Viewport(type, target);
	ui->gv->setScene(vp);
}

void DistViewGUI::initVC(representation::t type)
{
	/* create VC, apply UI to it, then add to GV */
	vc = new AutohideWidget();
	uivc = new Ui::ViewportControl();
	uivc->setupUi(vc);
	ui->gv->addWidget(AutohideWidget::LEFT, vc);
	// small hack to enable proximity trigger for said autohide widget
	ui->gv->fitContentRect(QRect(20, 0, 0, 0)); //!! change when adding widgets!

	/* we are ready to connect signals/slots */

	// let user see how many bins he will end up with
	connect(uivc->binSlider, SIGNAL(sliderMoved(int)),
	        this, SLOT(setBinLabel(int)));
	connect(uivc->binSlider, SIGNAL(valueChanged(int)),
	        this, SLOT(setBinCount(int)));
	connect(uivc->alphaSlider, SIGNAL(valueChanged(int)),
	        this, SLOT(setAlpha(int)));

	// low quality drawing while user works the slider
	connect(uivc->alphaSlider, SIGNAL(sliderPressed()),
	        vp, SLOT(startNoHQ()));
	connect(uivc->alphaSlider, SIGNAL(sliderReleased()),
	        vp, SLOT(endNoHQ()));

	connect(uivc->limiterButton, SIGNAL(toggled(bool)),
	        vp, SLOT(setLimitersMode(bool)));
	connect(uivc->limiterMenuButton, SIGNAL(clicked()),
	        this, SLOT(showLimiterMenu()));

	connect(uivc->rgbButton, SIGNAL(toggled(bool)),
	        vp, SLOT(toggleRGB(bool)));

	connect(uivc->screenshotButton, SIGNAL(released()),
	        vp, SLOT(screenshot()));

	connect(uivc->hqButton, SIGNAL(released()),
	        vp, SLOT(toggleHQ()));

	connect(uivc->formatButton, SIGNAL(released()),
	        this, SLOT(showFrameBufferMenu()));

	connect(uivc->logButton, SIGNAL(released()),
	        vp, SLOT(toggleDrawLog()));

	connect(uivc->rgbButton, SIGNAL(toggled(bool)),
	        this, SIGNAL(rgbToggled(bool)));

	setAlpha(uivc->alphaSlider->value());
	setBinCount(uivc->binSlider->value());

	createFrameBufferMenu();
}

void DistViewGUI::initTop()
{
	// connect toggling trigger
	connect(ui->topBar, SIGNAL(toggleFold()),
	        this, SLOT(toggleFold()));

	// setup title in topBar
	setTitle(type);
}

void DistViewGUI::initSignals(QObject *dvctrl)
{
	// signals from DistviewController
	connect(dvctrl, SIGNAL(pixelOverlayInvalid()),
	        vp, SLOT(removePixelOverlay()));
	connect(dvctrl, SIGNAL(toggleLabeled(bool)),
	        vp, SLOT(toggleLabeled(bool)));
	connect(dvctrl, SIGNAL(toggleUnlabeled(bool)),
	        vp, SLOT(toggleUnlabeled(bool)));

	connect(dvctrl, SIGNAL(labelSelected(int)),
	        vp, SLOT(toggleLabelHighlight(int)));

	// signals to controller
	connect(this, SIGNAL(requestBinCount(representation::t, int)),
	        dvctrl, SLOT(changeBinCount(representation::t, int)));


	//   viewport action
	connect(vp, SIGNAL(activated(representation::t)),
	        dvctrl, SLOT(setActiveViewer(representation::t)));
	connect(vp, SIGNAL(activated(representation::t)),
	        this, SIGNAL(activated()));
	connect(vp, SIGNAL(bandSelected(int)),
	        dvctrl, SLOT(propagateBandSelection(int)));
	connect(vp, SIGNAL(requestOverlay(int,int)),
	        dvctrl, SLOT(drawOverlay(int,int)));
	connect(vp, SIGNAL(requestOverlay(std::vector<std::pair<int,int> >,int)),
	        dvctrl, SLOT(drawOverlay(std::vector<std::pair<int,int> >,int)));

	connect(vp, SIGNAL(addSelectionRequested()),
	        dvctrl, SLOT(addHighlightToLabel()));
	connect(vp, SIGNAL(remSelectionRequested()),
	        dvctrl, SLOT(remHighlightFromLabel()));

	//    subscriptions
	connect(this, SIGNAL(subscribeRepresentation(QObject*,representation::t)),
	        dvctrl, SIGNAL(subscribeRepresentation(QObject*,representation::t)));
	connect(this, SIGNAL(unsubscribeRepresentation(QObject*,representation::t)),
	        dvctrl, SIGNAL(unsubscribeRepresentation(QObject*,representation::t)));


	// illumination correction
	connect(this, SIGNAL(newIlluminantCurve(QVector<multi_img::Value>)),
	        vp, SLOT(changeIlluminantCurve(QVector<multi_img::Value>)));
	connect(this, SIGNAL(toggleIlluminationShown(bool)),
	        vp, SLOT(setIlluminationCurveShown(bool)));
	connect(this, SIGNAL(newIlluminantApplied(QVector<multi_img::Value>)),
	        vp, SLOT(setAppliedIlluminant(QVector<multi_img::Value>)));
}

void DistViewGUI::initSubscriptions()
{
	if (!ui->gv->isHidden()) {
		emit needBinning(type);
		emit subscribeRepresentation(this, type);
	}
}

void DistViewGUI::setEnabled(bool enabled)
{
	// TODO: maybe do this on frame
	ui->gv->setEnabled(enabled);
}

void DistViewGUI::fold(bool folded)
{
	if (folded) { // fold
		GGDBGM(type << " folding" << endl);

		// let the controller know we do not need our image representation
		emit unsubscribeRepresentation(this, type);

		// fold GUI and set size policy for proper arrangement
		ui->gv->setHidden(true);
		ui->topBar->fold();
		frame->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);

		// reset title
		setTitle(type);

		// TODO: send signals that will destroy unused data!

		// TODO: let viewport clean itself up!
		//(*binsets)->clear(); // clear the vector, not re-create shareddata!
		//viewport->shuffleIdx.clear();
		//viewport->vb.destroy();
		emit foldingStateChanged(type, true);

	} else { // unfold
		GGDBGM(type << " unfolding" << endl);

		emit needBinning(type);
		// let the controller know we need our image representation
		emit subscribeRepresentation(this, type);

		// unfold GUI and set size policy for proper arrangement
		ui->gv->setVisible(true);
		ui->topBar->unfold();
		QSizePolicy pol(QSizePolicy::Preferred, QSizePolicy::Expanding);
		pol.setVerticalStretch(1);
		frame->setSizePolicy(pol);

		emit foldingStateChanged(type, false);

		// TODO: trigger calculation of data?
	}
}

void DistViewGUI::toggleFold()
{
	if (!ui->gv->isHidden()) {
		fold(true);
	} else {
		fold(false);
	}
}

void DistViewGUI::setTitle(representation::t type)
{
	QString title = QString("<b>%1</b>").arg(representation::prettyString(type));
	ui->topBar->setTitle(title);
}

void DistViewGUI::setTitle(representation::t type,
                           multi_img::Value min, multi_img::Value max)
{
	QString title = QString("<b>%1</b>").arg(representation::prettyString(type));
	title = title.append(" [%1..%2]")
	        .arg(min, 0, 'f', 2).arg(max, 0, 'f', 2);
	ui->topBar->setTitle(title);
}

void DistViewGUI::setAlpha(int alpha)
{
	float realalpha = (float)alpha/100.f;
	uivc->alphaLabel->setText(QString::fromUtf8("α: %1")
	                          .arg(realalpha, 0, 'f', 2));

	vp->setAlpha(realalpha);
}

void DistViewGUI::setBinLabel(int n)
{
	uivc->binLabel->setText(QString("%1 bins").arg(n));
}

void DistViewGUI::setBinCount(int n)
{
	setBinLabel(n);
	emit requestBinCount(type, n);
}

void DistViewGUI::updateLabelColors(QVector<QColor> colors)
{
	labelColors = colors;

	/* we need to rebuild label color values
	 * initial menu creation is implicitely done as updateLabelColors is called
	 * at startup */
	createLimiterMenu();
}

void DistViewGUI::createLimiterMenu()
{
	limiterMenu.clear();
	QAction *tmp;
	tmp = limiterMenu.addAction("No limits");
	tmp->setData(0);
	tmp = limiterMenu.addAction("Limit from current highlight");
	tmp->setData(-1);
	limiterMenu.addSeparator();
	for (int i = 1; i < labelColors.size(); ++i) {
		tmp = limiterMenu.addAction(colorIcon(labelColors.at(i)),
		                            "Limit by label");
		tmp->setData(i);
	}
}

void DistViewGUI::showLimiterMenu()
{
	// map to scene coordinates
	QPoint scenepoint = uivc->limiterMenuButton->mapToGlobal(QPoint(0, 0));
	// map to screen coordinates
	QPoint screenpoint = ui->gv->mapToGlobal(scenepoint);

	QAction *a = limiterMenu.exec(screenpoint);
	if (!a)
		return;

	int choice = a->data().toInt(); assert(choice < labelColors.size());
	vp->setLimiters(choice);
	if (!uivc->limiterButton->isChecked()) {
		uivc->limiterButton->toggle();	// changes button state AND viewport
	} else {
		vp->setLimitersMode(true);	// only viewport
	}
}

void DistViewGUI::createFrameBufferMenu()
{
	QActionGroup *actionGroup = new QActionGroup(this);
	actionGroup->setExclusive(true);

	QAction* tmp;
	tmp = frameBufferMenu.addAction("RGBA8");
	tmp->setCheckable(true);
	tmp->setData(Viewport::BufferFormat::RGBA8);
	actionGroup->addAction(tmp);

	tmp = frameBufferMenu.addAction("RGBA16F");
	tmp->setCheckable(true);
	tmp->setChecked(true);
	tmp->setData(Viewport::BufferFormat::RGBA16F);
	actionGroup->addAction(tmp);

	tmp = frameBufferMenu.addAction("RGBA32F");
	tmp->setCheckable(true);
	tmp->setData(Viewport::BufferFormat::RGBA32F);
	actionGroup->addAction(tmp);
}

void DistViewGUI::showFrameBufferMenu()
{
	// map to scene coordinates
	QPoint scenepoint = uivc->formatButton->mapToGlobal(QPoint(0, 0));
	// map to screen coordinates
	QPoint screenpoint = ui->gv->mapToGlobal(scenepoint);

	QAction *a = frameBufferMenu.exec(screenpoint);
	if (!a)
		return;

	Viewport::BufferFormat choice = (Viewport::BufferFormat) a->data().toInt();
	vp->setBufferFormat(choice);
}

void DistViewGUI::insertPixelOverlay(const QPolygonF &points)
{
	vp->insertPixelOverlay(points);
}

/** Return a 32x32px icon filled with color. */
QIcon DistViewGUI::colorIcon(const QColor &color)
{
	QPixmap pm(32, 32);
	pm.fill(color);
	return QIcon(pm);
}

bool DistViewGUI::isVisible()
{
	return !ui->gv->isHidden();
}
