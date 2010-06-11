#include "viewport.h"
#include "qpainter.h"
#include <iostream>
#include <QtCore>
#include <QPaintEvent>
#include <QRect>

using namespace std;

Viewport::Viewport(QWidget *parent)
	: QGLWidget(QGLFormat(QGL::SampleBuffers), parent),
	  selection(0), hover(-1), showLabeled(true), showUnlabeled(true),
	  active(false), zoom(1.), shift(0), lasty(-1)
{}

QTransform Viewport::getModelview()
{
	/* apply zoom and translation in window coordinates */
	qreal wwidth = width();
	qreal wheight = height()*zoom;
	int vshift = height()*shift;

	int p = 10; // padding
	// if gradient, we discard one unit space intentionally for centering
	int d = dimensionality - (gradient? 0 : 1);
	qreal w = (wwidth  - 2*p)/(qreal)(d); // width of one unit
	qreal h = (wheight - 2*p)/(qreal)(nbins - 1); // height of one unit
	int t = (gradient? w/2 : 0); // moving half a unit for centering

	QTransform modelview;
	modelview.translate(p + t, p + vshift);
	modelview.scale(w, -1*h); // -1 low values at bottom
	modelview.translate(0, -(nbins -1)); // shift for low values at bottom


	// cache it
	modelviewI = modelview.inverted();

	return modelview;
}

void Viewport::paintEvent(QPaintEvent *event)
{
	// return early if no data present. other variables may not be initialized
	if (sets.empty())
		return;

	QBrush background(QColor(15, 7, 15));
	QPainter painter(this);
	painter.fillRect(rect(), background);
	painter.setWorldTransform(getModelview());
	painter.setRenderHint(QPainter::Antialiasing);
	/* make sure that viewport shows "unlabeled" in the ignore label case */
	int start = ((showUnlabeled || ignoreLabels == 1) ? 0 : 1);
	int end = (showLabeled ? sets.size() : 1);
	for (int i = start; i < end; ++i) {
		BinSet &s = sets[i];
		QColor basecolor = s.label, color;
		QHash<QByteArray, Bin>::const_iterator it;
		for (it = s.bins.constBegin(); it != s.bins.constEnd(); ++it) {
			const Bin &b = it.value();
			color = basecolor;

			qreal alpha;
			/* TODO: this is far from optimal yet. challenge is to give a good
			   view where important information is not lost, yet not clutter
			   the view with too much low-weight information */
			if (i == 0)
				alpha = 0.01 + 0.09*(log(b.weight) / log(s.totalweight));
			else
				alpha = log(b.weight+1) / log(s.totalweight);
			color.setAlphaF(min(alpha, 1.));

			if ((unsigned char)it.key()[selection] == hover) {
				if (basecolor == Qt::white) {
					color = Qt::yellow;
				} else {
					color.setGreen(min(color.green() + 195, 255));
					color.setRed(min(color.red() + 195, 255));
					color.setBlue(color.blue()/2);
				}
				color.setAlphaF(1.);
			}

			painter.setPen(color);
			painter.drawLines(b.connections);
		}
	}
	if (active)
		painter.setPen(Qt::red);
	else
		painter.setPen(Qt::gray);
	painter.drawLine(selection, 0, selection, nbins);
}

void Viewport::mouseMoveEvent(QMouseEvent *event)
{
	if (event->buttons() & Qt::RightButton)
	{
		if (lasty < 0) {
			return;
		}

		shift += (event->y() - lasty)*0.05/(qreal)height();
		qreal displacement = (shift + 1./zoom);

		update();
	} else {
		QPoint pos = modelviewI.map(event->pos());
		int x = pos.x(), y = pos.y();
		if (x < 0 || x >= dimensionality)
			return;
		if (y < 0 || y >= nbins)
			return;

		if ((selection == x)&&(hover == y))
			return;

		if (selection != x)
			emit bandSelected(x, gradient);

		selection = x;
		hover = y;
		update();
		emit newOverlay();
	}
}

void Viewport::mousePressEvent(QMouseEvent *event)
{
	if (!active)
		emit activated(gradient);
	active = true;

	if (event->button() == Qt::RightButton) {
		this->setCursor(Qt::ClosedHandCursor);
		lasty = event->y();
	}
	mouseMoveEvent(event);
}

void Viewport::mouseReleaseEvent(QMouseEvent * event)
{
	if (event->button() == Qt::RightButton) {
		this->setCursor(Qt::ArrowCursor);
		lasty = -1;
	}
}

void Viewport::wheelEvent(QWheelEvent *event)
{
	qreal oldzoom = zoom;
	if (event->delta() > 0)
		zoom *= 1.25;
	else
		zoom = max(zoom * 0.80, 1.);

	// adjust shift to new zoom
	shift += ((oldzoom - zoom) * 0.5);

	update();
	event->accept();
}
