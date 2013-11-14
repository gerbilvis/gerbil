#include "widgets/autohidewidget.h"

#include <QGraphicsScene>

AutohideWidget::AutohideWidget()
	: location(LEFT), state(SCROLL_OUT)
{
	/* set dark, transparent background and corresponding white text */
	QPalette palette;
	QBrush brush(QColor(255, 255, 255, 255));
	brush.setStyle(Qt::SolidPattern);
	palette.setBrush(QPalette::Active, QPalette::WindowText, brush);
	QBrush brush1(QColor(250, 250, 250, 255));
	brush1.setStyle(Qt::SolidPattern);
	palette.setBrush(QPalette::Active, QPalette::Base, brush1);
	QBrush brush2(QColor(63, 63, 63, 191));
	brush2.setStyle(Qt::SolidPattern);
	palette.setBrush(QPalette::Active, QPalette::Window, brush2);
	palette.setBrush(QPalette::Inactive, QPalette::WindowText, brush);
	palette.setBrush(QPalette::Inactive, QPalette::Base, brush1);
	palette.setBrush(QPalette::Inactive, QPalette::Window, brush2);
	QBrush brush3(QColor(144, 144, 145, 255));
	brush3.setStyle(Qt::SolidPattern);
	palette.setBrush(QPalette::Disabled, QPalette::WindowText, brush3);
	palette.setBrush(QPalette::Disabled, QPalette::Base, brush2);
	palette.setBrush(QPalette::Disabled, QPalette::Window, brush2);
	setPalette(palette);
}

void AutohideWidget::init(QGraphicsProxyWidget *p, border loc)
{
	location = loc;
	proxy = p;

	/* move widget to the margin of the display
	 * so it will pop-out on demand
	 */
	switch (location) {
	case LEFT:
		proxy->setPos(OutOffset - width(), 0.f);
		break;
	case RIGHT:
		proxy->setPos(proxy->scene()->width() - OutOffset, 0.f);
		break;
	case TOP:
		proxy->setPos(0.f, OutOffset - height());
		break;
	case BOTTOM:
		proxy->setPos(0.f, proxy->scene()->height() - OutOffset);
		break;
	}
}

void AutohideWidget::adjustToSize(QSize size)
{
	switch (location) {
	case RIGHT:
		if (state == STAY_IN || state == SCROLL_IN)
			proxy->setPos(proxy->scene()->width() - width(), 0.f);
		else
			proxy->setPos(proxy->scene()->width() - OutOffset, 0.f);
		// no break here
	case LEFT:
		setMinimumHeight(size.height());
		setMaximumHeight(size.height());
		break;

	case BOTTOM:
		if (state == STAY_IN || state == SCROLL_IN)
			proxy->setPos(0.f, proxy->scene()->height() - height());
		else
			proxy->setPos(0.f, proxy->scene()->height() - OutOffset);
		// no break here
	case TOP:
		setMinimumWidth(size.width());
		setMaximumWidth(size.width());
		break;
	}
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
	bool proximity;
	switch (location) {
	case LEFT:
		proximity = (pos.x() < ownpos.x() + width() + OutOffset);
		break;
	case RIGHT:
		proximity = (pos.x() > ownpos.x() - OutOffset);
		break;
	case TOP:
		proximity = (pos.y() < ownpos.y() + height() + OutOffset);
		break;
	case BOTTOM:
		proximity = (pos.y() > ownpos.y() - OutOffset);
		break;
	}
	if (state == SCROLL_IN && !proximity)
		scrollOut();
	if (state == SCROLL_OUT && proximity)
		scrollIn();
}

void AutohideWidget::scrollOut(bool enforce)
{
	// already scrolling?
	bool redundant = (state == SCROLL_OUT || state == STAY_OUT);
	state = (enforce ? STAY_OUT : SCROLL_OUT);

	if (!redundant)
		startTimer(40); // scroll
}

void AutohideWidget::scrollIn(bool enforce)
{
	// already scrolling?
	bool redundant = (state == SCROLL_IN || state == STAY_IN);
	state = (enforce ? STAY_IN : SCROLL_IN);

	if (!redundant)
		startTimer(40); // scroll
}

void AutohideWidget::timerEvent(QTimerEvent *e)
{
	bool update = false;

	// position in the scene (as we have no parent)
	QPointF pos = proxy->pos();
	// position of the *outer* edge relative to scene border
	qreal relpos;
	// distance between outer and inner edge
	qreal offset;
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

	if (state == SCROLL_IN || state == STAY_IN) {
		// if not fully in scene, scroll further in
		if (relpos < 0.f) {
			// adjust diff up to fully scrolled-in state
			relpos = std::min(relpos + (qreal)40.f, (qreal)0.f);
			update = true;
		}
	} else {
		// if not (almost) fully out of scene, scroll further out
		if ((relpos + offset) > OutOffset) {
			// adjust diff down to almost hidden state
			relpos = std::max(relpos - (qreal)60.f, OutOffset - offset);
			update = true;
		}
	}

	if (update) {
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
