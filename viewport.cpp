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

void Viewport::paintEvent(QPaintEvent *event)
{
	QBrush background(QColor(15, 7, 15));
	QPainter painter;
	QTransform modelview;
	modelview.translate(10, 10);
	modelview.scale((width() - 20)/(qreal)(dimensionality-1),
					(height() - 20)/(qreal)nbins);
	painter.begin(this);

	painter.fillRect(rect(), background);
	painter.setWorldTransform(modelview);
	painter.setRenderHint(QPainter::Antialiasing);
	for (int i = 0; i < sets.size(); ++i) {
		const BinSet *s = sets[i];
		QColor color = s->label;
		QHash<qlonglong, Bin>::const_iterator it;
		for (it = s->bins.constBegin(); it != s->bins.constEnd(); ++it) {
			const Bin &b = it.value();
			qreal alpha = log(b.weight) / log(s->totalweight);
			color.setAlphaF(min(alpha, 1.));
			painter.setPen(color);
			painter.drawLines(b.connections);
		}
	}
	painter.end();
}
