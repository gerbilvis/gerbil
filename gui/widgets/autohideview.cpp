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

QGLWidget* AutohideView::init(bool mouseTrack)
{
	target = new QGLWidget(QGLFormat(QGL::SampleBuffers));
	setViewport(target);
	// TODO: needed? setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
	// TODO: should we cache the background?

	if (mouseTrack)
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

void AutohideView::resizeEvent(QResizeEvent *event)
{
	/* always resize the scene accordingly */
	if (scene())
		scene()->setSceneRect(QRect(QPoint(0, 0), event->size()));

	/* also let the widgets adjust */
	foreach (AutohideWidget* w, widgets)
		w->adjustToSize(event->size());

	QGraphicsView::resizeEvent(event);
}

void AutohideView::mouseMoveEvent(QMouseEvent *event)
{
	if (!suppressScroll) {
		foreach (AutohideWidget* w, widgets)
			w->triggerScrolling(event->pos());
	}

	QGraphicsView::mouseMoveEvent(event);
}

void AutohideView::leaveEvent(QEvent *event)
{
	// only scrollout if cursor really moved out (no popup menu etc.)
	if (!rect().contains(mapFromGlobal(QCursor::pos()))) {
		foreach (AutohideWidget* w, widgets)
			w->scrollOut();
	}

	QGraphicsView::leaveEvent(event);
}
