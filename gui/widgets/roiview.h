#ifndef ROIVIEW_H
#define ROIVIEW_H

#include "widgets/scaledview.h"
#include "widgets/sizegripitem/sizegripitem.h"
#include <QGraphicsItem>
#include <QGraphicsSceneEvent>
#include <QCursor>
#include <QApplication>

class BoundedRect : public QObject, public QGraphicsRectItem
{
	Q_OBJECT

public:
	BoundedRect() : lastcursor(0.f, 0.f) {}
	QRect getRect() { return rect().toRect(); }
	void adjustTo(QRectF box, bool internal);

signals:
	void newRect(QRectF rect);
	void newSelection(const QRect& roi);

protected:
	// implement such that we will get mouse grab
	void mousePressEvent(QGraphicsSceneMouseEvent *ev);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent *)
	{ lastcursor = QPointF(0.f, 0.f); QApplication::restoreOverrideCursor(); }
	void mouseMoveEvent(QGraphicsSceneMouseEvent *ev);

	QPointF lastcursor;
};

class ROIView : public ScaledView
{
	Q_OBJECT

public:
	ROIView();

	QRect roi() const { return rect->getRect(); }
	void setROI(QRect roi);
	virtual void setPixmap(QPixmap p);

	void setApplyAction(QAction* action) { applyAction = action; }
	void setResetAction(QAction* action) { resetAction = action; }

signals:
	void newSelection(const QRect& roi);

protected:
	void resizeEvent();
	QMenu* createContextMenu();

	// ROI rectangle, children of container
	BoundedRect *rect;
	// container of ROI rectangle, needed for proper coord. transformation
	QGraphicsRectItem *container;

	QAction* applyAction;
	QAction* resetAction;
};

struct BoundedRectResizer : public SizeGripItem::Resizer
{
	virtual void operator()(QGraphicsItem* item, const QRectF& rect)
	{
		BoundedRect* rectItem =	dynamic_cast<BoundedRect*>(item);
		if (rectItem)
			rectItem->adjustTo(rect, false);
	}
};

#endif // ROIVIEW_H
