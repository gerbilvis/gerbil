#include "widgets/autohidewidget.h"

#include <vole_config.h>
#include <boost/program_options.hpp>

#include <QGraphicsScene>
#include <QTimer>
#include <QPainter>
#include <QLayout>

#include <algorithm>

ENUM_MAGIC_CLS(AutohideWidget, border)

AutohideWidget::AutohideWidget()
	: location(LEFT), state(SCROLL_OUT), show_indicator(true), triggerOffset(0)
{
	QString style = "AutohideWidget { background: rgba(63, 63, 63, 191); } ";
	// warning: background: transparent is important, otherwise the stylesheet
	// does not apply to QCheckBox. Maybe due to qt bug or actual stylesheet
	// rules.
	style.append("QLabel, QRadioButton, QCheckBox "
				 "{ background: transparent; color: white; }");
#ifdef _WIN32 // windows progress bars show the text next to the bar
	style.append("QProgressBar { color: white; }");
#endif
	setStyleSheet(style);
}

void AutohideWidget::init(QGraphicsProxyWidget *p, border loc)
{
	location = loc;
	proxy = p;

	/* add margin and load icon for outoffset */
	const int addendum = OutOffset - 4;
	auto margins = layout()->contentsMargins();
	switch (location) {
	case LEFT:
		indicator.load(":/autohide/left");
		margins.setRight(margins.right() + addendum);
		break;
	case RIGHT:
		indicator.load(":/autohide/right");
		margins.setLeft(margins.left() + addendum);
		break;
	case TOP:
		indicator.load(":/autohide/top");
		margins.setBottom(margins.bottom() + addendum);
		break;
	case BOTTOM:
		indicator.load(":/autohide/bottom");
		margins.setTop(margins.top() + addendum);
		break;
	}
	layout()->setContentsMargins(margins);

	reposition();
	/* Idea: start with SCROLL_IN to show that we exist, then disappear later
	QTimer::singleShot(2000, this, SLOT(scrollOut())); */
}

void AutohideWidget::reposition()
{
	qreal sceneWidth = proxy->scene()->width();
	qreal sceneHeight = proxy->scene()->height();

	switch (location) {
	case RIGHT:
		if (state == STAY_IN || state == SCROLL_IN)
			proxy->setPos(sceneWidth - width(), 0.f);
		else
			proxy->setPos(sceneWidth - OutOffset, 0.f);
		setFixedHeight(sceneHeight);
		break;
	case LEFT:
		if (state == STAY_IN || state == SCROLL_IN)
			proxy->setPos(0.f, 0.f);
		else
			proxy->setPos(OutOffset - width(), 0.f);
		setFixedHeight(sceneHeight);
		break;

	case TOP:
		if (state == STAY_IN || state == SCROLL_IN)
			proxy->setPos(0.f, 0.f);
		else
			proxy->setPos(0.f, OutOffset - height());
		setFixedWidth(sceneWidth);
		break;
	case BOTTOM:
		if (state == STAY_IN || state == SCROLL_IN)
			proxy->setPos(0.f, sceneHeight - height());
		else
			proxy->setPos(0.f, sceneHeight - OutOffset);
		setFixedWidth(sceneWidth);
		break;
	}

	// now we know if we are fully scrolled out or not
	show_indicator = (state == STAY_OUT || state == SCROLL_OUT);

	repaint();
}

void AutohideWidget::triggerScrolling(QPoint pos)
{
	// invalid position means we lost the mouse
	if (pos.x() < 0) {
		if (state == SCROLL_IN) // we are inside, but not enforced
			scrollOut();
		return;
	}

	QPointF ownpos = proxy->pos();
	bool proximity = false;
	switch (location) {
	case LEFT:
		proximity = (pos.x() < ownpos.x() + width() + triggerOffset);
		break;
	case RIGHT:
		proximity = (pos.x() > ownpos.x() - triggerOffset);
		break;
	case TOP:
		proximity = (pos.y() < ownpos.y() + height() + triggerOffset);
		break;
	case BOTTOM:
		proximity = (pos.y() > ownpos.y() - triggerOffset);
		break;
	}
	if (state == SCROLL_IN && !proximity)
		scrollOut();
	if (state == SCROLL_OUT && proximity)
		scrollIn();
}

void AutohideWidget::adjust()
{
	// first rescale the widget
	adjustSize();
	// then make sure it stays visible
	reposition();
}

void AutohideWidget::scrollIn(bool enforce)
{
	// already scrolling?
	bool redundant = (state == SCROLL_IN || state == STAY_IN);
	state = (enforce ? STAY_IN : SCROLL_IN);

	if (!redundant)
		startTimer(33); // scroll with 30fps
}

void AutohideWidget::scrollOut(bool enforce)
{
	// already scrolling?
	bool redundant = (state == SCROLL_OUT || state == STAY_OUT);
	state = (enforce ? STAY_OUT : SCROLL_OUT);

	if (!redundant)
		QTimer::singleShot(250, this, SLOT(scrollOutNow()));
}

void AutohideWidget::scrollOutNow()
{
	// only do this if no other scrolling was triggered in-between
	if (state == SCROLL_OUT || state == STAY_OUT)
		startTimer(33); // scroll with 30fps
}

void AutohideWidget::paintEvent(QPaintEvent *e)
{
	if (show_indicator) {
		QPainter painter(this);
		switch (location) {
		case LEFT:
			painter.drawPixmap(rect().width() - indicator.width(),
							   (rect().height() - indicator.height()) / 2,
							   indicator);
			break;
		case RIGHT:
			painter.drawPixmap(0,
							   (rect().height() - indicator.height()) / 2,
							   indicator);
			break;
		case TOP:
			painter.drawPixmap((rect().width() - indicator.width()) / 2,
							   rect().height() - indicator.height(), indicator);
			break;
		case BOTTOM:
			painter.drawPixmap((rect().width() - indicator.width()) / 2,
							   0, indicator);
			break;
		}
	}
	QWidget::paintEvent(e);
}

void AutohideWidget::timerEvent(QTimerEvent *e)
{
	// position in the scene (as we have no parent)
	QPointF pos = proxy->pos();
	// position of the *outer* edge relative to scene border
	qreal relpos = 0.;
	// distance between outer and inner edge
	qreal offset = 0.;
	switch (location) {
	case LEFT:
		offset = width();
		relpos = pos.x();
		break;
	case RIGHT:
		offset = width();
		relpos = -(pos.x() + offset - proxy->scene()->width());
		break;
	case TOP:
		offset = height();
		relpos = pos.y();
		break;
	case BOTTOM:
		offset = height();
		relpos = -(pos.y() + offset - proxy->scene()->height());
		break;
	}

	bool changed = false;
	bool old_indicator = show_indicator;
	// by default hide the indicator
	show_indicator = false;

	if (state == SCROLL_IN || state == STAY_IN) {
		// if not fully in scene, scroll further in
		if (relpos < 0.f) {
			// adjust diff up to fully scrolled-in state
			relpos = std::min(relpos + (qreal)25.f, (qreal)0.f);
			changed = true;
		}
	} else {
		// if not (almost) fully out of scene, scroll further out
		if ((relpos + offset) > OutOffset) {
			// adjust diff down to almost hidden state
			relpos = std::max(relpos - (qreal)40.f, OutOffset - offset);
			changed = true;
		} else {
			// show indicator so the user will find us
			show_indicator = true;
		}
	}

	if (show_indicator != old_indicator)
		repaint(); // get rid of indicator before movement

	if (changed) {
		switch (location) {
		case LEFT:
			pos.setX(relpos);
			break;
		case RIGHT:
			pos.setX(proxy->scene()->width() - (relpos + offset));
			break;
		case TOP:
			pos.setY(relpos);
			break;
		case BOTTOM:
			pos.setY(proxy->scene()->height() - (relpos + offset));
			break;
		}

		proxy->setPos(pos);
	} else {
		killTimer(e->timerId()); // no more updates
	}
}

void AutohideWidget::changeEvent(QEvent *e)
{
	QWidget::changeEvent(e);
	switch (e->type()) {
	case QEvent::LanguageChange:
		// retranslateUi(this); TODO: to be handled by overload.
		break;
	default:
		break;
	}
}
