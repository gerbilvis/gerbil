#include "autohideview.h"

#include <QGLWidget>
#include <cassert>

AutohideView::AutohideView(QWidget *parent)
	: QGraphicsView(parent), suppressScroll(false)
{
	// avoid floating point exceptions, unreasonable shrinkage
	setMinimumSize(50, 50);
	updateGeometry();
}

QGLWidget* AutohideView::init()
{
	/* Note: we might need different formats in different views. In the future,
	 * make the format a parameter */
	QGLFormat format;
	format.setDepth(false); // we don't use depth buffer (save GPU memory)
	// note: QPainter uses Stencil Buffer, so it stays
	format.setSampleBuffers(false); // TODO: configurable!
	target = new QGLWidget(format);
	setViewport(target);
	// best for QGLWidget
	setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

	/* mouse tracking will implicitely switched on as soon as we have widgets
	 * in the view. we make it explicit to avoid confusion. */
	target->setMouseTracking(true);

	// the target belongs to us, but others might need access to it
	return target;
}

void AutohideView::addWidget(AutohideWidget::border loc, AutohideWidget *w)
{
	assert(scene());
	assert(!widgets.contains(loc));

	// add to scene
	QGraphicsProxyWidget *p = scene()->addWidget(w);

	// make this widget a texture in GL
	p->setCacheMode(QGraphicsItem::DeviceCoordinateCache);

	// for future use: make moveable dialog
	// proxy->setFlag(QGraphicsItem::ItemIsMovable);

	// initialize the widget
	w->init(p, loc);

	// add to our own reference
	widgets.insert(loc, w);
}

void AutohideView::updateSizeHint(QSize sizeHint)
{
	hint = sizeHint;
	updateGeometry();
}

void AutohideView::fitContentRect(QRect rect)
{
	/* let autohide widgets that completely fit-in stay in */
	foreach (AutohideWidget::border location, widgets.keys()) {
		AutohideWidget* w = widgets.value(location);
		AutohideWidget::scrollstate state = w->scrollState();
		if (state == AutohideWidget::STAY_OUT)
			continue;

		bool keepIn;
		switch (location) {
		case AutohideWidget::LEFT:
			keepIn = (rect.left() > w->width());
			break;
		case AutohideWidget::RIGHT:
			keepIn = (rect.right() < width() - w->width());
			break;
		case AutohideWidget::TOP:
			keepIn = (rect.top() > w->height());
			break;
		case AutohideWidget::BOTTOM:
			keepIn = (rect.bottom() < height() - w->height());
			break;
		}

		if (keepIn) {
			w->scrollIn(true);  // enforce staying in
		} else if (state == AutohideWidget::STAY_IN) {
			w->scrollOut(false); // revert previous enforcement
		}
	}
}

void AutohideView::resizeEvent(QResizeEvent *event)
{
	/* always resize the scene accordingly */
	if (scene())
		scene()->setSceneRect(QRect(QPoint(0, 0), event->size()));

	/* also let the widgets adjust */
	foreach (AutohideWidget* w, widgets)
		w->reposition();

	QGraphicsView::resizeEvent(event);
}

void AutohideView::mouseMoveEvent(QMouseEvent *event)
{
	if (!suppressScroll) {
		foreach (AutohideWidget* w, widgets)
			w->triggerScrolling(event->pos(), 10);
	}

	QGraphicsView::mouseMoveEvent(event);
}

void AutohideView::leaveEvent(QEvent *event)
{
	// only let them know if cursor really moved out (no popup menu etc.)
	if (!rect().contains(mapFromGlobal(QCursor::pos()))) {
		foreach (AutohideWidget* w, widgets)
			w->triggerScrolling(QPoint(-1, -1));
	}

	QGraphicsView::leaveEvent(event);
}
