#include "roiview.h"

#include <stopwatch.h>
#include <QPainter>
#include <QPaintEvent>
#include <iostream>

ROIView::ROIView(QWidget *parent)
	: ScaledView(parent), lockX(-1), lockY(-1), lastcursor(-1, -1)
{}

void ROIView::paintEvent(QPaintEvent *ev)
{
	if (!pixmap)
		return;

	QPainter painter(this);
	painter.setRenderHint(QPainter::SmoothPixmapTransform);

	painter.setWorldTransform(scaler);
	QRect damaged = scalerI.mapRect(ev->rect());
	painter.drawPixmap(damaged, *pixmap, damaged);

	QColor color(Qt::gray); color.setAlpha(127);
	painter.fillRect(0, 0, roi.x(), pixmap->height(), color);
	painter.fillRect(roi.x()+roi.width(), 0,
					 pixmap->width() - roi.x() - roi.width(),
					 pixmap->height(), color);

	painter.fillRect(roi.x(), 0, roi.width(), roi.y(), color);
	painter.fillRect(roi.x(), roi.y()+roi.height(),
					 roi.width(), pixmap->height() - roi.y() - roi.height(),
					 color);
	painter.setPen(Qt::red);
	painter.drawRect(roi);
}

void ROIView::cursorAction(QMouseEvent *ev, bool click)
{
	QPointF cursor(ev->pos() / scale);
	cursor.setX(round(cursor.x() - 0.75));
	cursor.setY(round(cursor.y() - 0.75));

	// nothing new after all..
	if (cursor == lastcursor)
		return;

	int x = cursor.x(), y = cursor.y();
	x = std::max(std::min(x, pixmap->width() - 1), 0);
	y = std::max(std::min(y, pixmap->height() - 1), 0);


	bool right;
	if (roi.width() < 2)
		right = x > roi.x();
	else if (lockX > -1)
		right = (lockX > 0);
	else
		right = (std::abs(roi.x()-x) > std::abs(roi.right()-x));

	if (right)
		roi.setRight(x);
	else
		roi.setLeft(x);
	lockX = (right ? 1 : 0);

	bool bottom;
	if (roi.height() < 2)
		bottom = y > roi.y();
	else if (lockY > -1)
		bottom = (lockY > 0);
	else
		bottom = (std::abs(roi.y()-y) > std::abs(roi.bottom()-y));

	if (bottom)
		roi.setBottom(y);
	else
		roi.setTop(y);
	lockY = (bottom ? 1 : 0);

	update();
}

void ROIView::mouseReleaseEvent(QMouseEvent *)
{
	// enable 'grabbing' of different border on next click
	lockX = lockY = -1;
}
