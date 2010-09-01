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
	float w, h;
	if (src_aspect > dest_aspect) {
		w = width(); h = w/src_aspect;
	} else {
		h = height(); w = h*src_aspect;
	}
	scale = w/pixmap->width();
	scaler = QTransform().scale(scale, scale);
	scalerI = scaler.inverted();
}

void ScaledView::paintEvent(QPaintEvent *ev)
{
	QPainter painter(this);
	if (!pixmap) {
		painter.fillRect(this->rect(), QColor(Qt::lightGray));
		return;
	}

	painter.setRenderHint(QPainter::SmoothPixmapTransform);

	painter.setWorldTransform(scaler);
	QRect damaged = scalerI.mapRect(ev->rect());
	painter.drawPixmap(damaged, *pixmap, damaged);
}

void ScaledView::cursorAction(QMouseEvent *ev, bool click)
{
}

