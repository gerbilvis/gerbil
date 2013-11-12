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
#include <QPaintEvent>
#include <iostream>

/* TODO: do we really want sample buffers for these views? configurable?
 */
ScaledView::ScaledView()
	: width(50), height(50) // values don't matter much, but should be over 0
{}

QSize ScaledView::updateSizeHint()
{
	float src_aspect = 1.f;
	if (!pixmap.isNull())
		src_aspect = pixmap.width()/(float)pixmap.height();
	emit newSizeHint(QSize(300*src_aspect, 300));
}

void ScaledView::setPixmap(QPixmap p)
{
	pixmap = p;
	resizeEvent();
	updateSizeHint();
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
		w = (width - 3) - 1;
	else
		w = (height - 3)*src_aspect - 1;

	/* centering */
	scaler = QTransform().translate((width - w)/2.f,
									(height - w/src_aspect)/2.f);
	/* scaling */
	scale = w/pixmap.width();
	scaler.scale(scale, scale);

	// inverted transform to handle input
	scalerI = scaler.inverted();
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

void ScaledView::mouseMoveEvent(QGraphicsSceneMouseEvent *ev)
{
	cursorAction(ev);
}

void ScaledView::mousePressEvent(QGraphicsSceneMouseEvent *ev)
{
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
