#include "autohideview.h"

#include <QApplication>
#include <QTimer>
#include <QGLWidget>
#include <QWheelEvent>

#include <cassert>
#include <stdexcept>
#include <algorithm>

//#define GGDBG_MODULE
#include <gerbil_gui_debug.h>

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
	for (auto location : widgets.keys()) {
		AutohideWidget* w = widgets.value(location);
		AutohideWidget::scrollstate state = w->scrollState();
		if (state == AutohideWidget::STAY_OUT)
			continue;

		int offset; // offset of content to the relevant border
		bool keepIn;
		switch (location) {
		case AutohideWidget::LEFT:
			offset = rect.left();
			keepIn = (offset > w->width());
			break;
		case AutohideWidget::RIGHT:
			offset = width() - rect.right();
			keepIn = (offset > w->width());
			break;
		case AutohideWidget::TOP:
			offset = rect.top();
			keepIn = (offset > w->height());
			break;
		case AutohideWidget::BOTTOM:
			offset = height() - rect.bottom();
			keepIn = (offset > w->height());
			break;
		default:
			throw std::runtime_error("bad location in "
			                         "AutohideView::fitContentRect()");
		}

		// whole widget fits in whitespace
		if (keepIn) {
			w->scrollIn(true);  // enforce staying in
		} else if (state == AutohideWidget::STAY_IN) {
			w->scrollOut(false); // revert previous enforcement
		}

		// trigger scrollIn from widgets proximity, but not from inside content
		offset = std::min(0 + AutohideWidget::OutOffset,
		                  offset - AutohideWidget::OutOffset);
		offset = std::max(offset, 0);
		w->setTriggerOffset(offset);
	}
}

void AutohideView::resizeEvent(QResizeEvent *event)
{
	/* always resize the scene accordingly */
	if (scene())
		scene()->setSceneRect(QRect(QPoint(0, 0), event->size()));

	/* also let the widgets adjust */
	for (auto w : widgets)
		w->reposition();

	QGraphicsView::resizeEvent(event);
}

void AutohideView::mouseMoveEvent(QMouseEvent *event)
{
	if (!suppressScroll) {
		for (auto w : widgets)
			w->triggerScrolling(event->pos());
	}

	QGraphicsView::mouseMoveEvent(event);
}

void AutohideView::leaveEvent(QEvent *event)
{
	triggerScollOut();
	// Call triggerScollOut() again because the modal dialog test does
	// not work when we get the leave event -- the dialog is not active yet.
	// When the timer finishes, the dialog will be active or the mouse
	// does or does not point at us. All cases are correctly handled.
	QTimer::singleShot(500, this, SLOT(triggerScollOut()));

	QGraphicsView::leaveEvent(event);
}

void AutohideView::triggerScollOut()
{
	// FIXME: Not sure if this works on all platforms. On Linux
	// QApplication::activeWindow() == 0 if there is a modal dialog open.
	const bool haveModalWindow = QApplication::activeWindow() == 0 ||
	                             QApplication::activeWindow()->isModal();
	const bool cursorInsideView = rect().contains(mapFromGlobal(QCursor::pos()));
	const bool trigger = haveModalWindow || !cursorInsideView;

	GGDBGM("windows: this " << this
	       << ", active " << QApplication::activeWindow()
	       << ", trigger " << trigger << " <- "
	       << " haveModalWindow " << haveModalWindow
	       << " || "
	       << " !cursorInsideView " << !cursorInsideView
	       << endl);

	// only let them know if a modal dialog opened or
	// cursor really moved out (no popup menu etc.).
	if (trigger) {
		for (auto w : widgets) {
			GGDBGM("trigger " << w->getLocation() << endl);
			w->triggerScrolling(QPoint(-1, -1));
		}
	}

}

void AutohideView::applyCursor(Qt::CursorShape shape)
{
	curShape = shape;
	target->setCursor(shape);
}

