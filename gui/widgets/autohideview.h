#ifndef AUTOHIDEVIEW_H
#define AUTOHIDEVIEW_H

#include "autohidewidget.h"

#include <QGraphicsView>
#include <QMap>

class AutohideView : public QGraphicsView
{
	Q_OBJECT
public:
	AutohideView(QWidget *parent);

	/* provide a reasonably high size of correct aspect ratio for layouting */
	virtual QSize sizeHint() const {
		return QSize(1000, 200);
	}

	void addWidget(AutohideWidget::border location, AutohideWidget *widget);

public slots:

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

	// all scrolling widgets
	QMap<AutohideWidget::border, AutohideWidget*> widgets;

	// don't trigger scrolling in some situations
	bool suppressScroll;
};

#endif // AUTOHIDEVIEW_H
