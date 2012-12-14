/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "scaledview.h"

#include <stopwatch.h>
#include <QPainter>
#include <QPaintEvent>
#include <iostream>

ScaledView::ScaledView(QWidget *parent)
	: QGLWidget(QGLFormat(QGL::SampleBuffers), parent), pixmap(NULL)
{}

void ScaledView::setPixmap(const QPixmap &p)
{
	pixmap = &p;
	resizeEvent(0);
	update();
}

void ScaledView::resizeEvent(QResizeEvent *ev)
{
	if (!pixmap)
		return;

	// determine scale of correct aspect-ratio
	float src_aspect = pixmap->width()/(float)pixmap->height();
	float dest_aspect = width()/(float)height();
	float w;	// new width
	if (src_aspect > dest_aspect)
		w = width() - 1;
	else
		w = height()*src_aspect - 1;

	scale = w/pixmap->width();
	scaler = QTransform().scale(scale, scale);
	scalerI = scaler.inverted();
}

void ScaledView::paintEvent(QPaintEvent *ev)
{
	QPainter painter(this);
	if (!pixmap) {
		painter.fillRect(this->rect(), QBrush(Qt::gray, Qt::BDiagPattern));
		drawWaitMessage(painter);
		return;
	}

	painter.save();

	painter.setRenderHint(QPainter::SmoothPixmapTransform);

	painter.setWorldTransform(scaler);
	QRect damaged = scalerI.mapRect(ev->rect());
	painter.drawPixmap(damaged, *pixmap, damaged);

	painter.restore();

	if (!isEnabled()) {
		drawWaitMessage(painter);
	}
}

void ScaledView::drawWaitMessage(QPainter &painter)
{
	painter.save();
	// darken
	painter.fillRect(rect(), QColor(0, 0, 0, 127));

	// text in larger size with nice color
	painter.setPen(QColor(255, 230, 0));
	QFont tmp(font());
	tmp.setPointSize(tmp.pointSize() * 1.75);
	painter.setFont(tmp);
	painter.drawText(rect(), Qt::AlignCenter,
					 QString::fromUtf8("Calculating…"));
	painter.restore();
}

void ScaledView::cursorAction(QMouseEvent *ev, bool click)
{
}

