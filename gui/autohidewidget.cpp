#include "autohidewidget.h"

AutohideWidget::AutohideWidget()
	: state(STATE_OUT)
{
}

void AutohideWidget::initProxy(QGraphicsProxyWidget *i)
{
	proxy = i;
	proxy->setWidget(this);

	//proxy->setFlag(QGraphicsItem::ItemIsMovable);

	// make this widget a texture in GL
	proxy->setCacheMode(QGraphicsItem::DeviceCoordinateCache);

	/* move widget to the left margin of the display
	 * so it will pop-out on demand
	 */
	const QRectF rect = proxy->boundingRect();
	proxy->setPos(10.f - rect.width(), 0.f);
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
	QPointF pos = proxy->pos();
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
		proxy->setPos(pos);
//		std::cerr << proxy->pos().x() + width() << std::endl;
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
