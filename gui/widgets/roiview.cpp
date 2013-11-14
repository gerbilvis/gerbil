/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "widgets/roiview.h"

#include <stopwatch.h>
#include <QPainter>
#include <QGraphicsSceneEvent>
#include <iostream>
#include <cmath>

ROIView::ROIView()
	: lockX(-1), lockY(-1), lastcursor(-1, -1)
{}

void ROIView::paintEvent(QPainter *painter, const QRectF &rect)
{
	if (pixmap.isNull())
		return;

	fillBackground(painter, rect);

	painter->setRenderHint(QPainter::SmoothPixmapTransform);

	painter->setWorldTransform(scaler);
	QRectF damaged = scalerI.mapRect(rect);
	painter->drawPixmap(damaged, pixmap, damaged);

	/* gray out region outside ROI */
	QColor color(Qt::darkGray); color.setAlpha(127);
	painter->fillRect(0, 0, roi.x(), pixmap.height(), color);
	painter->fillRect(roi.x()+roi.width(), 0,
					 pixmap.width() - roi.x() - roi.width(),
					 pixmap.height(), color);

	painter->fillRect(roi.x(), 0, roi.width(), roi.y(), color);
	painter->fillRect(roi.x(), roi.y()+roi.height(),
					 roi.width(), pixmap.height() - roi.y() - roi.height(),
					 color);
	painter->setPen(Qt::red);
	painter->drawRect(roi);
}

void ROIView::cursorAction(QGraphicsSceneMouseEvent *ev, bool)
{
	// only perform an action when left mouse button pressed
	if (!(ev->buttons() & Qt::LeftButton))
		return;

	QPointF cursor = scalerI.map(QPointF(ev->scenePos()));
	cursor.setX(std::floor(cursor.x() - 0.25));
	cursor.setY(std::floor(cursor.y() - 0.25));

	// nothing new after all..
	if (cursor == lastcursor)
		return;

	int x = cursor.x(), y = cursor.y();
	x = std::max(std::min(x, pixmap.width() - 1), 0);
	y = std::max(std::min(y, pixmap.height() - 1), 0);


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
	emit newSelection(roi);
}

void ROIView::mouseReleaseEvent(QGraphicsSceneMouseEvent *ev)
{
	// check for scene elements first (we are technically the background)
	QGraphicsScene::mouseReleaseEvent(ev);
	if (ev->isAccepted())
		return;

	// enable 'grabbing' of different border on next click
	lockX = lockY = -1;
}
