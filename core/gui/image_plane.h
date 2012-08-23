#ifndef IMAGE_PLANE_H
#define IMAGE_PLANE_H

#include <QLabel>
#include <QMenu>
#include <QList>
#include <QPaintEvent>
#include <QPainter>

#include "rectangle.h"
#include "draw_operation.h"
#include "mouse_move_operation.h"

#include <iostream>

namespace Ui
{
class ImagePlane;
}

class ImagePlane : public QLabel
{
	Q_OBJECT

protected slots:
	void    zoomIn(void);
	void    zoomOut(void);
	void    addRectangle(int);
	void    removeRectangle();

public:
	ImagePlane(QWidget *parent = 0);
	void    setImage(QImage);
	void 	addRectangle(QRect, QString, QString);
	QRect   getRect(int);
	void    setPopMenu(QMenu*);

	void addPaintEventNotification(DrawOperation *);
	void addMouseMoveEventNotification(MouseMoveOperation *);
	void addMousePressEventNotification(MouseMoveOperation *);
	void addMouseReleaseEventNotification(MouseMoveOperation *);
	void addMouseDragEventNotification(MouseMoveOperation *);

	void removePaintEventNotification(DrawOperation *);
	void removeMouseMoveEventNotification(MouseMoveOperation *);
	void removeMousePressEventNotification(MouseMoveOperation *);
	void removeMouseReleaseEventNotification(MouseMoveOperation *);
	void removeMouseDragEventNotification(MouseMoveOperation *);


	virtual ~ImagePlane();

	QList<Rectangle*> &getRectList();

protected:
	QList<Rectangle*> rectList;
	QImage       img;
	QMenu *popMenu;
	bool         keyCTRL;
	double       scaFak;
	QPoint       mevPoint;


	virtual void paintEvent(QPaintEvent *);
	virtual void mouseMoveEvent(QMouseEvent *);
	virtual void mousePressEvent(QMouseEvent *);
	virtual void mouseReleaseEvent(QMouseEvent *);
	virtual void dragMoveEvent(QMouseEvent *);
	bool    eventFilter(QObject*, QEvent*);

	void    ScaleImage(double);

	void    wheelEvent(QWheelEvent*);
	void    keyPressEvent(QKeyEvent*);
	void    keyReleaseEvent(QKeyEvent*);

	std::vector<DrawOperation *> drawOperations;
	std::vector<MouseMoveOperation *> mouseMoveEvents;
	std::vector<MouseMoveOperation *> mousePressEvents;
	std::vector<MouseMoveOperation *> mouseReleaseEvents;
	std::vector<MouseMoveOperation *> mouseDragEvents;
};

#endif // IMAGE_PLANE_H
