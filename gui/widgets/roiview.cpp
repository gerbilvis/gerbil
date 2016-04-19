/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "widgets/roiview.h"
#include "widgets/sizegripitem/sizegripitem.h"

#include <stopwatch.h>
#include <QPainter>
#include <QGraphicsSceneEvent>
#include <iostream>
#include <cmath>
#include <algorithm>

ROIView::ROIView()
{
	// prevent panning/moving through ScaledView
	inputMode = InputMode::Disabled;

	rect = new BoundedRect();
	rect->setBrush(QColor(255, 255, 255, 31));
	QPen pen(Qt::DashLine);
	pen.setColor(Qt::lightGray);
	rect->setPen(pen);

	container = new QGraphicsRectItem();
	container->setFlag(QGraphicsItem::ItemHasNoContents);
	container->setBrush(Qt::NoBrush);
	container->setPen(Qt::NoPen);
	addItem(container);
	rect->setParentItem(container);

	SizeGripItem* grip = new SizeGripItem(new BoundedRectResizer, rect);

	connect(rect, SIGNAL(newRect(QRectF)), grip, SLOT(setRect(QRectF)));
	connect(rect, SIGNAL(newSelection(QRect)),
			this, SIGNAL(newSelection(QRect)));
}

void ROIView::setROI(QRect roi)
{
	rect->adjustTo(roi, true);
}

void ROIView::setPixmap(QPixmap p)
{
	container->setRect(p.rect());
	ScaledView::setPixmap(p);
}

void ROIView::resizeEvent()
{
	ScaledView::resizeEvent();
	container->setTransform(scaler);
	container->setRect(pixmap.rect());
}

QMenu *ROIView::createContextMenu()
{
	QMenu* contextMenu = ScaledView::createContextMenu();
	contextMenu->clear();

	contextMenu->addAction(applyAction);
	contextMenu->addAction(resetAction);

	return contextMenu;
}

void BoundedRect::adjustTo(QRectF box, bool internal)
{
	/* discretize */
	box.setLeft(std::floor(box.left()));
	box.setRight(std::floor(box.right()));
	box.setTop(std::floor(box.top()));
	box.setBottom(std::floor(box.bottom()));

	QRectF bound = parentItem()->boundingRect();
	/* adjustments needed */
	if (!bound.contains(box)) {
		// movement or rescaling? (handled differently)
		bool movement = box.size() == rect().size();
		if (movement) {
			box.moveLeft(std::max(box.left(), bound.left()));
			box.moveRight(std::min(box.right(), bound.right()));
			box.moveTop(std::max(box.top(), bound.top()));
			box.moveBottom(std::min(box.bottom(), bound.bottom()));
		} else {
			box.setLeft(std::max(box.left(), bound.left()));
			box.setRight(std::min(box.right(), bound.right()));
			box.setTop(std::max(box.top(), bound.top()));
			box.setBottom(std::min(box.bottom(), bound.bottom()));
		}
	}

	/* set rectangle and propagate */
	setRect(box);
	emit newRect(box);

	if (!internal)
		emit newSelection(box.toRect());
}

void BoundedRect::mouseMoveEvent(QGraphicsSceneMouseEvent *ev)
{
	// discretize cursor
	QPointF cursor;
	cursor.setX(std::floor(ev->pos().x() - 0.25));
	cursor.setY(std::floor(ev->pos().y() - 0.25));

	// initial movement
	if (lastcursor.isNull())
		lastcursor = cursor;

	// nothing new after all..
	if (cursor == lastcursor)
	  return;

	/* translate into our coordinates. Note: the way we do it, mouse movement
	 * and item movement are in perfect sync. if we would use Qt's functionality
	 * (ItemIsMovable), sadly the scaling would not fit. */
	QPointF diff = transform().map(cursor - lastcursor);

	// apply difference
	QRectF box = rect();
	box.translate(diff);

	// try to adjust
	adjustTo(box, false);

	// remember where we took off
	lastcursor = cursor;
}

void BoundedRect::mousePressEvent(QGraphicsSceneMouseEvent *ev)
{
	if (ev->button() == Qt::RightButton) {
		//ignore right button
		QGraphicsRectItem::mousePressEvent(ev);
	} else if (ev->button() == Qt::LeftButton) {
		lastcursor = QPointF(0.f, 0.f);
		QApplication::setOverrideCursor(QCursor(Qt::SizeAllCursor));
	}
}
