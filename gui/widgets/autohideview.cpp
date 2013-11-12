#include "autohideview.h"

#include <cassert>

AutohideView::AutohideView(QWidget *parent)
 : QGraphicsView(parent), suppressScroll(false)
{
}

void AutohideView::addWidget(AutohideWidget::border loc, AutohideWidget *w)
{
	assert(scene());
	assert(!widgets.find(loc));

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
	foreach (AutohideWidget* w, widgets)
		w->scrollOut();

	QGraphicsView::leaveEvent(event);
}
