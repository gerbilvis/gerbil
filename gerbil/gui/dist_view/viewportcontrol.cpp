#include "viewportcontrol.h"
#include "viewport.h"
#include "mainwindow.h"

#include <QGraphicsItem>

ViewportControl::ViewportControl(multi_img_viewer *parent) :
	holder(parent), viewport(0),
	limiterMenu((QWidget*)holder), // menu needs parent outside GraphicsScene
	state(STATE_OUT)
{
	setupUi(this);
}

void ViewportControl::init(Viewport *vp)
{
	viewport = vp;

	/* we are ready to connect signals/slots */
	connect(binSlider, SIGNAL(valueChanged(int)),
			(QWidget*)holder, SLOT(changeBinCount(int)));
	connect(binSlider, SIGNAL(valueChanged(int)),
			this, SLOT(setBinCount(int)));
	connect(alphaSlider, SIGNAL(valueChanged(int)),
			this, SLOT(setAlpha(int)));
	connect(alphaSlider, SIGNAL(sliderPressed()),
			viewport, SLOT(startNoHQ()));
	connect(alphaSlider, SIGNAL(sliderReleased()),
			viewport, SLOT(endNoHQ()));
	connect(limiterButton, SIGNAL(toggled(bool)),
			(QWidget*)holder, SLOT(toggleLimiters(bool)));
	connect(limiterMenuButton, SIGNAL(clicked()),
			this, SLOT(showLimiterMenu()));

	connect(rgbButton, SIGNAL(toggled(bool)),
			viewport, SLOT(toggleRGB(bool)));

	// default UI stuff
	setAlpha(alphaSlider->value());
	setBinCount(binSlider->value());
}

void ViewportControl::setType(representation::t type)
{
	if (type != representation::IMG)
		rgbButton->setVisible(false);
}

void ViewportControl::setAlpha(int alpha)
{
	viewport->useralpha = (float)alpha/100.f;
	alphaLabel->setText(QString::fromUtf8("α: %1").arg(viewport->useralpha, 0, 'f', 2));
	viewport->updateTextures(Viewport::RM_STEP, Viewport::RM_SKIP);
}

void ViewportControl::setBinCount(int n)
{
	binLabel->setText(QString("%1 bins").arg(n));
}

void ViewportControl::updateLabelColors(QVector<QColor> colors)
{
	labelColors = colors;

	/* we need to rebuild label color values
	 * initial menu creation is implicitely done as updateLabelColors is called
	 * at startup by selectROI */
	limiterMenu.clear();
	QAction *tmp;
	tmp = limiterMenu.addAction("No limits");
	tmp->setData(0);
	tmp = limiterMenu.addAction("Limit from current highlight");
	tmp->setData(-1);
	limiterMenu.addSeparator();
	for (int i = 1; i < labelColors.size(); ++i) {
		tmp = limiterMenu.addAction(MainWindow::colorIcon(labelColors[i]),
													  "Limit by label");
		tmp->setData(i);
	}
}

void ViewportControl::showLimiterMenu()
{
	// map to scene coordinates
	QPoint scenepoint = limiterMenuButton->mapToGlobal(QPoint(0, 0));
	// map to screen coordinates
	QPoint screenpoint = viewport->target->mapToGlobal(scenepoint);

	QAction *a = limiterMenu.exec(screenpoint);
	if (!a)
		return;

	int choice = a->data().toInt(); assert(choice < labelColors.size());
	viewport->setLimiters(choice);
	if (!limiterButton->isChecked()) {
		limiterButton->toggle();	// changes button state AND toggleLimiters()
	} else {
		holder->toggleLimiters(true);		// only toggleLimiters()
	}
}

void ViewportControl::scrollOut()
{
	// already scrolling
	if (state == STATE_OUT)
		return;

	state = STATE_OUT;
	startTimer(40);
}

void ViewportControl::scrollIn()
{
	// already scrolling
	if (state == STATE_IN)
		return;

	state = STATE_IN;
	startTimer(50);
}

void ViewportControl::timerEvent(QTimerEvent *e)
{
	bool update = false;
	QPointF pos = viewport->controlItem->pos();
	if (state == STATE_IN) {
		if (pos.x() < 0.f) {
			pos.setX(std::min(pos.x() + (qreal)40.f, (qreal)0.f));
			update = true;
		}
	} else {
		if (pos.x() > 10.f - width()) {
			pos.setX(std::max(pos.x() - (qreal)60.f, (qreal)(10.f - width())));
			update = true;
		}
	}

	if (update) {
		viewport->controlItem->setPos(pos);
//		std::cerr << viewport->controlItem->pos().x() + width() << std::endl;
	} else {
		killTimer(e->timerId()); // no more updates
	}
}

void ViewportControl::changeEvent(QEvent *e)
{
	QWidget::changeEvent(e);
	switch (e->type()) {
	case QEvent::LanguageChange:
		retranslateUi(this);
		break;
	default:
		break;
	}
}
