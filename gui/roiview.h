#ifndef ROIVIEW_H
#define ROIVIEW_H

#include "scaledview.h"

#include <cv.h>

class ROIView : public ScaledView
{
	Q_OBJECT
public:
	ROIView(QWidget *parent = 0);

	void paintEvent(QPaintEvent *ev);

	QRect roi;
private:
	void cursorAction(QMouseEvent *ev, bool click = false);
};

#endif // ROIVIEW_H
