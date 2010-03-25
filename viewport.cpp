#include "viewport.h"
#include "qpainter.h"
#include <iostream>
#include <QtCore>
#include <QPaintEvent>
#include <QRect>

using namespace std;

Viewport::Viewport(QWidget *parent)
	: QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
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
	modelview.scale(w, h);
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
		QColor color = s->label;
		QHash<QByteArray, Bin>::const_iterator it;
		for (it = s->bins.constBegin(); it != s->bins.constEnd(); ++it) {
			const Bin &b = it.value();
			qreal alpha = 0.01 + 0.99*(log(b.weight) / log(s->totalweight));
			color.setAlphaF(min(alpha, 1.));
			painter.setPen(color);
			painter.drawLines(b.connections);
		}
	}
	painter.end();
}
