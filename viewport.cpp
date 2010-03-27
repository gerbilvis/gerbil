#include "viewport.h"
#include "qpainter.h"
#include <iostream>
#include <QtCore>
#include <QPaintEvent>
#include <QRect>

using namespace std;

Viewport::Viewport(QWidget *parent)
	: QGLWidget(QGLFormat(QGL::SampleBuffers), parent),
	  selection(0), hover(-1)
{}

QTransform Viewport::getModelview()
{
	int p = 10; // padding
	// if gradient, we discard one unit space intentionally for centering
	int d = dimensionality - (gradient? 0 : 1);
	qreal w = (width()  - 2*p)/(qreal)(d); // width of one unit
	qreal h = (height() - 2*p)/(qreal)(nbins - 1); // height of one unit
	int t = (gradient? w/2 : 0); // moving half a unit for centering

	QTransform modelview;
	modelview.translate(p + t, p);
	modelview.scale(w, -1*h); // -1 low values at bottom
	modelview.translate(0, -(nbins -1));
	return modelview;
}

void Viewport::paintEvent(QPaintEvent *event)
{
	// return early if no data present. other variables may not be initialized
	if (sets.empty())
		return;

	QBrush background(QColor(15, 7, 15));
	QPainter painter;
	painter.begin(this);
	painter.fillRect(rect(), background);
	painter.setWorldTransform(getModelview());
	painter.setRenderHint(QPainter::Antialiasing);
	for (int i = 0; i < sets.size(); ++i) {
		const BinSet *s = sets[i];
		QColor basecolor = s->label, color;
		QHash<QByteArray, Bin>::const_iterator it;
		for (it = s->bins.constBegin(); it != s->bins.constEnd(); ++it) {
			const Bin &b = it.value();
			color = basecolor;

			qreal alpha = 0.01 + 0.99*(log(b.weight) / log(s->totalweight));
			color.setAlphaF(min(alpha, 1.));

			if ((unsigned char)it.key()[selection] == hover) {
				color.setBlue(0);
				color.setAlphaF(1.);
			}

			painter.setPen(color);
			painter.drawLines(b.connections);
		}
	}
	painter.setPen(Qt::red);
	painter.drawLine(selection, 0, selection, nbins);
	painter.end();
}

void Viewport::mouseMoveEvent(QMouseEvent *event)
{
	QPoint pos = getModelview().inverted().map(event->pos());
	int x = pos.x(), y = pos.y();
	if (x < 0 || x >= dimensionality)
		return;

	selection = x;
	hover = pos.y();
	repaint();
	emit sliceSelected(x, gradient);
}

SliceLabel::SliceLabel(QWidget *parent) : QLabel(parent)
{}

void SliceLabel::paintEvent(QPaintEvent *event)
{
	const QPixmap *p = pixmap();
	float src_aspect = p->width()/(float)p->height();
	float dest_aspect = width()/(float)height();
	float w, h;
	if (src_aspect > dest_aspect) {
		w = width(); h = w/src_aspect;
	} else {
		h = height(); w = h*src_aspect;
	}

	QPainter painter;
	painter.begin(this);
	painter.drawPixmap(0, 0, w, h, *p);
	painter.end();
}
