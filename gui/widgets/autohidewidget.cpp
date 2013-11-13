#include "widgets/autohidewidget.h"

#include <QGraphicsScene>

AutohideWidget::AutohideWidget()
	: location(LEFT), state(STATE_OUT)
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
		// reset to sunken state instead of messing with differential movement
		proxy->setPos(proxy->scene()->width() - OutOffset, 0.f);
		state = STATE_OUT;
		// no break here
	case LEFT:
		setMinimumHeight(size.height());
		break;

	case BOTTOM:
		// reset to sunken state instead of messing with differential movement
		proxy->setPos(0.f, proxy->scene()->height() - OutOffset);
		state = STATE_OUT;
		// no break here
	case TOP:
		setMinimumWidth(size.width());
		break;
	}
}

void AutohideWidget::triggerScrolling(QPoint pos)
{
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
	if (state == STATE_IN && !proximity)
		scrollOut();
	if (state == STATE_OUT && proximity)
		scrollIn();
}

void AutohideWidget::scrollOut()
{
	// already scrolling
	if (state == STATE_OUT)
		return;

	state = STATE_OUT;
	startTimer(40);
}

void AutohideWidget::scrollIn()
{
	// already scrolling
	if (state == STATE_IN)
		return;

	state = STATE_IN;
	startTimer(50);
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

	if (state == STATE_IN) {
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
