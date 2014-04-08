#ifndef AUTOHIDEVIEW_H
#define AUTOHIDEVIEW_H

#include "autohidewidget.h"

#include <QGraphicsView>
#include <QMap>
#include <iostream>

class QGLWidget;

class AutohideView : public QGraphicsView
{
	Q_OBJECT
public:
	explicit AutohideView(QWidget *parent);

	QGLWidget* init();

	/* provide a reasonably high size of correct aspect ratio for layouting
	 * Note: to make this work correctly with Qt's layouting habits, set the
	 * baseSize property of this widget, such that it is known early enough
	 */
	virtual QSize sizeHint() const {
		return (hint.isEmpty() ? baseSize() : hint);
	}

	void addWidget(AutohideWidget::border location, AutohideWidget *widget);

public slots:

	// provide newer size hint
	void updateSizeHint(QSize sizeHint);
	// optimize autohide widgets for bounding rectangle of the content
	void fitContentRect(QRect rect);
	// explicitely (un)lock scroll suppression
	void suppressScrolling(bool suppress) { suppressScroll = suppress; }

protected:

	// adjust scene and widgets
	void resizeEvent(QResizeEvent *event);

	// scroll in/out widgets
	void mouseMoveEvent(QMouseEvent *event);

	// scroll out widgets
	void leaveEvent(QEvent *event);

	// don't scroll through dragging operations
	void mousePressEvent(QMouseEvent *event)
	{ suppressScroll = true; QGraphicsView::mousePressEvent(event); }
	void mouseReleaseEvent(QMouseEvent *event)
	{ suppressScroll = false; QGraphicsView::mouseReleaseEvent(event); }

	// our target widget (we keep it as viewport() only returns a QWidget*)
	QGLWidget *target;

	// size hint to provide when asked
	QSize hint;

	// all scrolling widgets
	QMap<AutohideWidget::border, AutohideWidget*> widgets;

	// don't trigger scrolling in some situations
	bool suppressScroll;
};

#endif // AUTOHIDEVIEW_H
