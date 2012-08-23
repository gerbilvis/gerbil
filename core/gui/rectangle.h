#ifndef VOLE_RECTANGLE_H
#define VOLE_RECTANGLE_H

#include <QLabel>
#include <QWidget>

namespace Ui
{
class Rectangle;
}

namespace vole {
	enum RectangularPosition {
		NO_POSITION,
		BOTTOM_LEFT,
		BOTTOM,
		BOTTOM_RIGHT,
		LEFT,
		CENTER,
		RIGHT,
		TOP_LEFT,
		TOP,
		TOP_RIGHT
	};
}

class Rectangle : public QLabel
{
	Q_OBJECT

public:
	Rectangle(QWidget *parent = 0, QRect *rect = 0, int bdPix = 1,
		 QColor bdCol = Qt::black, QColor bgCol = QColor(255, 0, 0, 127),
		 bool sq = false, int mouseGrabRange = 3);
	virtual ~Rectangle();

protected:
	void    paintEvent(QPaintEvent*);
	bool    eventFilter(QObject*, QEvent*);
	bool	checkRegion(
				QRegion reg,
				QPoint &mousePos,
				vole::RectangularPosition currentPosition);
	void ensureMinimumSize(QPoint &delta);
	void ensureImageBoundaries(QPoint &delta, QPoint& p1, QPoint& p2);
	bool topGrab();
	bool bottomGrab();
	bool leftGrab();
	bool rightGrab();
	bool isVerticalDirection();
	bool isHorizontalDirection();
	bool isDiagonalDirection();

private:
	int    bdPix;
	QColor bdColor;        // border color
	QColor bgColor;        // background color
	bool   sqFlag;         // rectangle should be a square

// moving and resizing
	QPoint lastMousePos;
	bool   taskMoveRectangle, taskResizeRectangle;
	vole::RectangularPosition grabPos;
	int mouseGrabRange;


};

#endif // VOLE_RECTANGLE_H
