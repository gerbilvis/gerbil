/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "scaledview.h"

#include <stopwatch.h>
#include <QGLWidget>
#include <QPainter>
#include <QGraphicsSceneEvent>
#include <QDebug>

#include <iostream>

/* TODO: do we really want sample buffers for these views? configurable?
 */
ScaledView::ScaledView()
	: width(50), height(50), // values don't matter much, but should be over 0
	  zoom(1), sm(Zoom)
{
	// by default small offsets; can be altered from outside
	offLeft = offTop = offRight = offBottom = 2;
}

void ScaledView::updateSizeHint()
{
	float src_aspect = 1.f;
	if (!pixmap.isNull())
		src_aspect = pixmap.width()/(float)pixmap.height();
	emit newSizeHint(QSize(300*src_aspect, 300));
}

void ScaledView::setPixmap(QPixmap p)
{
	pixmap = p;

	bool cond = (p.width() != pixmap.width()
							  || p.height() != pixmap.height());
	if (cond) {
		resizeEvent();
		updateSizeHint();
	}
}

void ScaledView::drawBackground(QPainter *painter, const QRectF &rect)
{
	// update geometry
	int nwidth = painter->device()->width();
	int nheight = painter->device()->height();
	if (nwidth != width || nheight != height) {
		width = nwidth;
		height = nheight;
		resizeEvent();
	}

	// paint
	paintEvent(painter, rect);
}

void ScaledView::resizeEvent()
{
	if (pixmap.isNull())
		return;

	// determine scale of correct aspect-ratio
	float src_aspect = pixmap.width()/(float)pixmap.height();
	float dest_aspect = width/(float)height;
	float w;	// new width
	if (src_aspect > dest_aspect)
		w = (width - offLeft - offRight);
	else
		w = (height - offTop - offBottom)*src_aspect;

	/* centering */
	scaler.reset();
	scaler.translate(offLeft + (width - offLeft - offRight - w)/2.f,
					 offTop + (height - offTop - offBottom - w/src_aspect)/2.f);
	/* scaling */
	float scale = w/pixmap.width();
	scaler.scale(scale, scale);

	// inverted transform to handle input
	scalerI = scaler.inverted();

	// let the view know about the geometry we actually do occupy
	emit newContentRect(scaler.mapRect(pixmap.rect()));
}

void ScaledView::paintEvent(QPainter *painter, const QRectF &rect)
{
	if (!pixmap) {
		painter->fillRect(rect, QBrush(Qt::gray, Qt::BDiagPattern));
		drawWaitMessage(painter);
		return;
	}

	fillBackground(painter, rect);

	painter->save();

	painter->setRenderHint(QPainter::SmoothPixmapTransform);
	painter->setWorldTransform(scaler);
	QRectF damaged = scalerI.mapRect(rect);
	painter->drawPixmap(damaged, pixmap, damaged);

	painter->restore();
}

void ScaledView::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	QGraphicsScene::mouseMoveEvent(event);
	cursorAction(event);

	if (sm != Zoom) return;

	if (event->buttons() == Qt::LeftButton) {
		//Obtain current cursor and last cursor position
		//in pixmap coordinates
		QPointF lastonscene = scalerI.map(event->lastScenePos());
		QPointF curronscene = scalerI.map(event->scenePos());

		qreal x = curronscene.x() - lastonscene.x();
		qreal y = curronscene.y() - lastonscene.y();

		scaler.translate(x,y);
		scalerI = scaler.inverted();
	}
}

void ScaledView::mousePressEvent(QGraphicsSceneMouseEvent *ev)
{
	// check for scene elements first (we are technically the background)
	QGraphicsScene::mousePressEvent(ev);
	if (ev->isAccepted())
		return;

	cursorAction(ev, true);
}

void ScaledView::drawWaitMessage(QPainter *painter)
{
	painter->save();
	// darken
	painter->fillRect(sceneRect(), QColor(0, 0, 0, 127));

	// text in larger size with nice color
	painter->setPen(QColor(255, 230, 0));
	QFont tmp(font());
	tmp.setPointSize(tmp.pointSize() * 1.75);
	painter->setFont(tmp);
	painter->drawText(sceneRect(), Qt::AlignCenter,
					  QString::fromUtf8("Calculating…"));
	painter->restore();
}

void ScaledView::cursorAction(QGraphicsSceneMouseEvent *ev, bool click)
{
}

void ScaledView::wheelEvent(QGraphicsSceneWheelEvent *event)
{
	QGraphicsScene::wheelEvent(event);

	if (sm != Zoom) return;

	qreal newzoom;

	if (event->delta() > 0) {
		newzoom = 1.25;
	} else {
		newzoom = 0.8;
	}

	if (zoom*newzoom < 1) {
		resizeEvent();
	} else {
		//obtain cursor position in scene coordinates
		QPointF scene = event->scenePos();
		//obtain cursor position in pixmap coordinates
		QPointF local = scalerI.map(scene);

		zoom *= newzoom;
		//scaling
		scaler.scale(newzoom, newzoom);
		scalerI = scaler.inverted();

		//after scaling there's different point under cursor
		//so we have to obtain cursor position in pixmap coordinates
		//once again
		QPointF newlocal = scalerI.map(scene);

		//translate the by the difference
		QPointF diff = newlocal - local;
		scaler.translate(diff.x(), diff.y());
		scalerI = scaler.inverted();
	}
}
