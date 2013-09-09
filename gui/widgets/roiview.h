#ifndef ROIVIEW_H
#define ROIVIEW_H

#include "widgets/scaledview.h"


class ROIView : public ScaledView
{
	Q_OBJECT
public:
	ROIView(QWidget *parent = 0);

	void paintEvent(QPaintEvent *ev);
	void mouseReleaseEvent(QMouseEvent*);

	QRect roi;

signals:
	void newSelection(const QRect& roi);

private:
	void cursorAction(QMouseEvent *ev, bool click = false);

	int lockX, lockY;
	QPointF lastcursor;
};

#endif // ROIVIEW_H
