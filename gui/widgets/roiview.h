#ifndef ROIVIEW_H
#define ROIVIEW_H

#include "widgets/scaledview.h"


class ROIView : public ScaledView
{
	Q_OBJECT
public:
	ROIView();

	void paintEvent(QPainter *painter, const QRectF &rect);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent*);

	QRect roi;

signals:
	void newSelection(const QRect& roi);

private:
	void cursorAction(QGraphicsSceneMouseEvent *ev, bool click = false);

	int lockX, lockY;
	QPointF lastcursor;
};

#endif // ROIVIEW_H
