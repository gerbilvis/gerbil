#include "distviewgui.h"
#include "controller/distviewcontroller.h"
#include "widgets/autohidewidget.h"

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

	// default UI stuff
	if (type != representation::IMG)
		uivc->rgbButton->setVisible(false);

	setAlpha(uivc->alphaSlider->value());
	setBinCount(uivc->binSlider->value());
}

void DistViewGUI::initTop()
{
	// connect toggling trigger
	connect(ui->topBar, SIGNAL(toggleFold()),
			this, SLOT(toggleFold()));

	// setup title in topBar
	setTitle(type);
}

void DistViewGUI::initSignals(DistViewController *chief)
{
	// signals from controller
	connect(chief, SIGNAL(pixelOverlayInvalid()),
			 vp, SLOT(removePixelOverlay()));
	connect(chief, SIGNAL(toggleLabeled(bool)),
			vp, SLOT(toggleLabeled(bool)));
	connect(chief, SIGNAL(toggleUnlabeled(bool)),
			vp, SLOT(toggleUnlabeled(bool)));

	connect(chief, SIGNAL(singleLabelSelected(int)),
			vp, SLOT(highlightSingleLabel(int)));

	// signals to controller
	connect(this, SIGNAL(requestBinCount(representation::t, int)),
			chief, SLOT(changeBinCount(representation::t, int)));

	/* propagate folding signal from our dist view to ALL of them,
	 * including us (back-connection) */
	connect(this, SIGNAL(folding()),
			chief, SIGNAL(folding()));

	// viewport action
	connect(vp, SIGNAL(activated(representation::t)),
			chief, SLOT(setActiveViewer(representation::t)));
	connect(vp, SIGNAL(activated(representation::t)),
			this, SIGNAL(activated()));
	connect(vp, SIGNAL(bandSelected(int)),
			chief, SLOT(propagateBandSelection(int)));
	connect(vp, SIGNAL(requestOverlay(int,int)),
			chief, SLOT(drawOverlay(int,int)));
	connect(vp, SIGNAL(requestOverlay(std::vector<std::pair<int,int> >,int)),
			chief, SLOT(drawOverlay(std::vector<std::pair<int,int> >,int)));

	connect(vp, SIGNAL(addSelectionRequested()),
			chief, SLOT(addHighlightToLabel()));
	connect(vp, SIGNAL(remSelectionRequested()),
			chief, SLOT(remHighlightFromLabel()));


	// illumination correction
	connect(this, SIGNAL(newIlluminantCurve(QVector<multi_img::Value>)),
			vp, SLOT(changeIlluminantCurve(QVector<multi_img::Value>)));
	connect(this, SIGNAL(toggleIlluminationShown(bool)),
			vp, SLOT(setIlluminationCurveShown(bool)));
	connect(this, SIGNAL(newIlluminantApplied(QVector<multi_img::Value>)),
			vp, SLOT(setAppliedIlluminant(QVector<multi_img::Value>)));
}

void DistViewGUI::setEnabled(bool enabled)
{
	// TODO: maybe do this on frame
	ui->gv->setEnabled(enabled);
}

void DistViewGUI::toggleFold()
{
	emit folding();

	if (!ui->gv->isHidden()) {
		/** HIDE **/

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
	} else {
		/** SHOW **/

		// unfold GUI and set size policy for proper arrangement
        ui->gv->setVisible(true);
		ui->topBar->unfold();
		QSizePolicy pol(QSizePolicy::Preferred, QSizePolicy::Expanding);
		pol.setVerticalStretch(1);
		frame->setSizePolicy(pol);

		// TODO: trigger calculation of data?
	}
}

void DistViewGUI::setTitle(representation::t type)
{
	QString title = QString("<b>%1</b>").arg(representation::str(type));
	ui->topBar->setTitle(title);
}

void DistViewGUI::setTitle(representation::t type,
						   multi_img::Value min, multi_img::Value max)
{
	QString title = QString("<b>%1</b>").arg(representation::str(type));
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

void DistViewGUI::insertPixelOverlay(const QPolygonF &points)
{
	vp->insertPixelOverlay(points);
}

void DistViewGUI::toggleSingleLabel(bool toggle)
{
	if (!toggle) {
		// disable single label highlight
		vp->highlightSingleLabel(-1);
	} // nothing to do else
}

/** Return a 32x32px icon filled with color. */
QIcon DistViewGUI::colorIcon(const QColor &color)
{
	QPixmap pm(32, 32);
	pm.fill(color);
	return QIcon(pm);
}
