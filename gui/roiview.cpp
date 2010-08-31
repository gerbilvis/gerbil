#include "roiview.h"

#include <stopwatch.h>
#include <QPainter>
#include <QPaintEvent>
#include <iostream>

ROIView::ROIView(QWidget *parent)
	: ScaledView(parent)
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
}

