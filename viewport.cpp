#include "viewport.h"
#include "qpainter.h"
#include <iostream>
#include <QtCore>
#include <QPaintEvent>
#include <QRect>

using namespace std;

Viewport::Viewport(QWidget *parent)
	: QGLWidget(QGLFormat(QGL::SampleBuffers), parent),
	  selection(0), hover(-1), showLabeled(true), showUnlabeled(true)
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
	QPainter painter(this);
	painter.fillRect(rect(), background);
	painter.setWorldTransform(getModelview());
	painter.setRenderHint(QPainter::Antialiasing);
	int start = (showUnlabeled ? 0 : 1);
	int end = (showLabeled ? sets.size() : 1);
	for (int i = start; i < end; ++i) {
		BinSet &s = sets[i];
		QColor basecolor = s.label, color;
		QHash<QByteArray, Bin>::const_iterator it;
		for (it = s.bins.constBegin(); it != s.bins.constEnd(); ++it) {
			const Bin &b = it.value();
			color = basecolor;

			qreal alpha = 0.01 + 0.99*(log(b.weight) / log(s.totalweight));
			color.setAlphaF(min(alpha, 1.));

			if ((unsigned char)it.key()[selection] == hover) {
				if (basecolor == Qt::white)
					color = Qt::yellow;
				color.setAlphaF(1.);
			}

			painter.setPen(color);
			painter.drawLines(b.connections);
		}
	}
	painter.setPen(Qt::red);
	painter.drawLine(selection, 0, selection, nbins);
}

void Viewport::mouseMoveEvent(QMouseEvent *event)
{	// todo: inefficient! cache the inverted transform
	QPoint pos = getModelview().inverted().map(event->pos());
	int x = pos.x(), y = pos.y();
	if (x < 0 || x >= dimensionality)
		return;

	selection = x;
	hover = pos.y();
	update();
	emit sliceSelected(x, gradient);
}

SliceView::SliceView(QWidget *parent)
	: QLabel(parent), cursor(-1, -1), lastcursor(-1, -1), curLabel(1), cacheValid(false)
{
	markerColors << Qt::white // 0 is index for unlabeled
			<< Qt::green << Qt::red << Qt::cyan << Qt::magenta << Qt::blue;
}

void SliceView::setPixmap(const QPixmap &p)
{
	cacheValid = false;
	QLabel::setPixmap(p);
}

void SliceView::resizeEvent(QResizeEvent *ev)
{
	// determine scale of correct aspect-ratio
	const QPixmap *p = pixmap();
	float src_aspect = p->width()/(float)p->height();
	float dest_aspect = width()/(float)height();
	float w, h;
	if (src_aspect > dest_aspect) {
		w = width(); h = w/src_aspect;
	} else {
		h = height(); w = h*src_aspect;
	}
	scale = w/p->width();
	scaler = QTransform().scale(scale, scale);
	scalerI = scaler.inverted();
}

void SliceView::paintEvent(QPaintEvent *ev)
{
	if (!cacheValid)
		updateCache();

	QPainter painter(this);
	painter.setRenderHint(QPainter::Antialiasing);

	// draw slice (slow!)
	painter.drawPixmap(ev->rect(), cachedPixmap.transformed(scaler), ev->rect());

	painter.setWorldTransform(scaler);
	// draw slice (artifacts)
/*	QRect damaged = scalerI.mapRect(ev->rect());
	painter.drawPixmap(damaged, cachedPixmap, damaged);*/

	// draw current cursor
	QPen pen(markerColors[curLabel]);
	pen.setWidth(0);
	painter.setPen(pen);
	painter.drawRect(QRectF(cursor, QSizeF(1, 1)));
}

void SliceView::updateCache()
{
	cachedPixmap = pixmap()->copy(); // TODO: check for possible qt memory leak
	QPixmap *p = &cachedPixmap;
	{	QPainter painter(p);

		// mark labeled regions
		for (int y = 0; y < p->height(); ++y) {
			for (int x = 0; x < p->width(); ++x) {
				int l = labels->pixelIndex(x, y);
				if (l > 0) {
					//painter.setBrush();
					QColor col = markerColors[l];
					col.setAlphaF(0.5);
					painter.setPen(col);
					painter.drawPoint(x, y);
				}
			}
		}
	}
	cacheValid = true;
}

void SliceView::mouseMoveEvent(QMouseEvent *ev)
{
	cursor = QPointF(ev->pos() / scale);
	cursor.setX(round(cursor.x() - 0.75));
	cursor.setY(round(cursor.y() - 0.75));
	int x = cursor.x(), y = cursor.y();

	if (!pixmap()->rect().contains(x, y))
		return;

	// paint
	if (ev->buttons() & Qt::LeftButton) {
		labels->setPixel(x, y, curLabel);
		cacheValid = false;
	// erase
	} else if (ev->buttons() & Qt::RightButton) {
		if (labels->pixelIndex(x, y) == curLabel)
			labels->setPixel(x, y, 0);
		cacheValid = false; // TODO: improve by altering cache directly
		updatePoint(cursor);
	}

	updatePoint(lastcursor);
	lastcursor = cursor;
}

void SliceView::updatePoint(const QPointF &p)
{
	QPoint damagetl = scaler.map(QPoint(p.x() - 1, p.y() - 1));
	QPoint damagebr = scaler.map(QPoint(p.x() + 1, p.y() + 1));
	update(QRect(damagetl, damagebr));
}

void SliceView::clearLabelPixels()
{
	for (int y = 0; y < labels->height(); ++y) {
		for (int x = 0; x < labels->width(); ++x) {
			if (labels->pixelIndex(x, y) == curLabel)
				labels->setPixel(x, y, 0);
		}
	}
	cacheValid = false;
	update();
}

void SliceView::leaveEvent(QEvent *ev)
{
	cursor = QPoint(-1, -1);
	update();
}

void SliceView::changeLabel(int label)
{
	if (label > -1)
		curLabel = label + 1; // we start with 1, combobox with 0
}
