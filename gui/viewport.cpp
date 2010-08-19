#include "viewport.h"
#include "qpainter.h"
#include <iostream>
#include <cmath>
#include <QtCore>
#include <QPaintEvent>
#include <QRect>

using namespace std;

Viewport::Viewport(QWidget *parent)
	: QGLWidget(QGLFormat(QGL::SampleBuffers), parent),
	  illuminant(NULL), selection(0), hover(-1), limiterMode(false),
	  active(false), wasActive(false), useralpha(1.f),
	  showLabeled(true), showUnlabeled(true), ignoreLabels(false),
	  overlayMode(false),
	  zoom(1.), shift(0), lasty(-1), cacheValid(false),
	  holdSelection(false)
{}

void Viewport::updateModelview()
{
	/* apply zoom and translation in window coordinates */
	qreal wwidth = width();
	qreal wheight = height()*zoom;
	int vshift = height()*shift;

	int hp = 20, vp = 10; // horizontal and vertical padding
	int tp = 20; // lower padding for text (legend)
	// if gradient, we discard one unit space intentionally for centering
	int d = dimensionality - (gradient? 0 : 1);
	qreal w = (wwidth  - 2*hp)/(qreal)(d); // width of one unit
	qreal h = (wheight - 2*vp - tp)/(qreal)(nbins - 1); // height of one unit
	int t = (gradient? w/2 : 0); // moving half a unit for centering

	modelview.reset();
	modelview.translate(hp + t, vp + vshift);
	modelview.scale(w, -1*h); // -1 low values at bottom
	modelview.translate(0, -(nbins -1)); // shift for low values at bottom

	// set inverse
	modelviewI = modelview.inverted();
}

void Viewport::drawBins(QPainter &painter)
{
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
			/* logarithm is used to prevent single data points to get lost.
			   this should be configurable. */
			if (i == 0)
				alpha = useralpha *
						(0.01 + 0.09*(log(b.weight+1) / log(s.totalweight)));
			else
				alpha = useralpha *
						(log(b.weight+1) / log(s.totalweight));
			color.setAlphaF(min(alpha, 1.));

			bool highlighted = false;
			QByteArray K = it.key();
			if (limiterMode) {
				highlighted = true;
				for (int i = 0; i < dimensionality; ++i) {
					unsigned char k = K[i];
					if (k < limiters[i].first || k > limiters[i].second)
						highlighted = false;
				}
			} else if ((unsigned char)K[selection] == hover)
				highlighted = true;

			if (highlighted) {
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
}

void Viewport::drawAxesFg(QPainter &painter)
{
	if (selection < 0 || selection >= dimensionality)
		return;

	// draw selection in foreground
	if (active)
		painter.setPen(Qt::red);
	else
		painter.setPen(Qt::gray);
	qreal top = (nbins-1);
	if (illuminant)
		top *= illuminant->at(selection);
	painter.drawLine(QPointF(selection, 0.), QPointF(selection, top));

	// draw limiters
	if (limiterMode) {
		painter.setPen(Qt::red);
		for (int i = 0; i < dimensionality; ++i) {
			qreal h = nbins*0.01;
			qreal y1 = limiters[i].first, y2 = limiters[i].second;
			QPolygonF polygon;
			polygon << QPointF(i - 0.25, y1 + h)
					<< QPointF(i - 0.25, y1)
					<< QPointF(i + 0.25, y1)
					<< QPointF(i + 0.25, y1 + h);
			painter.drawPolyline(polygon);
			polygon.clear();
			polygon << QPointF(i - 0.25, y2 - h)
					<< QPointF(i - 0.25, y2)
					<< QPointF(i + 0.25, y2)
					<< QPointF(i + 0.25, y2 - h);
			painter.drawPolyline(polygon);
		}
	}
}
void Viewport::drawAxesBg(QPainter &painter)
{
	// draw axes in background
	painter.setPen(QColor(64, 64, 64));
	QPolygonF poly;
	if (illuminant) {
		for (int i = 0; i < dimensionality; ++i) {
			qreal top = (nbins-1) * illuminant->at(i);
			painter.drawLine(QPointF(i, 0.), QPointF(i, top));
			poly << QPointF(i, top);
		}
		poly << QPointF(dimensionality-1, nbins-1);
		poly << QPointF(0, nbins-1);
	} else {
		for (int i = 0; i < dimensionality; ++i)
			painter.drawLine(i, 0, i, nbins-1);
	}

	// visualize illuminant
	if (illuminant) {
		QPolygonF poly2 = modelview.map(poly);
		poly2.translate(0., -5.);
		painter.restore();
		QBrush brush(QColor(32, 32, 32), Qt::Dense3Pattern);
		painter.setBrush(brush);
		painter.setPen(Qt::NoPen);
		painter.drawPolygon(poly2);
		painter.setPen(Qt::white);
		poly2.remove(dimensionality, 2);
		painter.drawPolyline(poly2);
		painter.save();
		painter.setWorldTransform(modelview);
	}
}

void Viewport::drawLegend(QPainter &painter)
{
	assert(labels.size() == (unsigned int)dimensionality);

	painter.setPen(Qt::white);
	for (int i = 0; i < dimensionality; ++i) {
		QPointF l = modelview.map(QPointF(i - 1.f, 0.f));
		QPointF r = modelview.map(QPointF(i + 1.f, 0.f));
		QRectF rect(l, r);
		rect.setHeight(30.f);

		// only draw every second label if we run out of space
		if (i % 2 && rect.width() < 50)
			continue;

		if (i == selection)
			painter.setPen(Qt::red);
		painter.drawText(rect, Qt::AlignCenter, labels[i]);
		if (i == selection)	// revert back color
			painter.setPen(Qt::white);
	}
}

void Viewport::drawRegular()
{
	QPainter painter(this);
	QBrush background(QColor(15, 7, 15));
	painter.fillRect(rect(), background);
	painter.setRenderHint(QPainter::Antialiasing);

	drawLegend(painter);

	painter.save();
	painter.setWorldTransform(modelview);

	drawAxesBg(painter);
	drawBins(painter);
	drawAxesFg(painter);

	painter.restore();
}

void Viewport::drawOverlay()
{
	QPainter painter(this);
	painter.drawImage(0, 0, cacheImg);
	painter.setRenderHint(QPainter::Antialiasing);

	QPen pen(Qt::black);
	pen.setWidth(5);
	painter.setPen(pen);
	for (int i = 0; i < overlayLines.size(); ++i) {
		painter.drawLine(modelview.map(overlayLines[i]));
	}
	QPen pen2(Qt::yellow);
	pen2.setWidth(2);
	painter.setPen(pen2);
	for (int i = 0; i < overlayLines.size(); ++i) {
		painter.drawLine(modelview.map(overlayLines[i]));
	}
}

void Viewport::paintEvent(QPaintEvent *event)
{
	// return early if no data present. other variables may not be initialized
	if (sets.empty())
		return;

	if (!overlayMode) {
		drawRegular();
		cacheValid = false;
		return;
	}

	// we draw an overlay. check cache first
	if (!cacheValid) {
		cacheImg = this->grabFrameBuffer();
		cacheValid = true;
	}

	drawOverlay();
}

void Viewport::resizeEvent(QResizeEvent *)
{
	updateModelview();
}

void Viewport::updateXY(int sel, int bin)
{
	bool emitOverlay = !wasActive;

	/// first handle sel -> band selection
	if (sel < 0 || sel >= dimensionality)
		return;

	/* emit new band if either new selection or first input after
	   activating this view */
	if ((selection != sel && !holdSelection) || !wasActive) {
		wasActive = true;
		emit bandSelected(sel, gradient);
	}

	/// second handle bin -> intensity highlight
	if (illuminant)	/* correct y for illuminant */
		bin = std::floor(bin / illuminant->at(sel) + 0.5f);

	if (bin < 0 || bin >= nbins)
		return;

	/// do the final update
	if (selection != sel && !holdSelection) {
		selection = sel;
		emitOverlay = true;
		if (limiterMode)
			holdSelection = true;
	}
	if (!limiterMode && (hover != bin)) {
		hover = bin;
		emitOverlay = true;
	}
	if (limiterMode && updateLimiter(selection, bin))
		emitOverlay = true;

	if (emitOverlay) {
		update();	/* TODO: check for dual update! */
		emit newOverlay();
	}
}

void Viewport::mouseMoveEvent(QMouseEvent *event)
{
	if (event->buttons() & Qt::RightButton) // panning movement
	{
		if (lasty < 0)
			return;

		shift += (event->y() - lasty)/(qreal)height();
		lasty = event->y();

		/* TODO: make sure that we use full visible space */

		updateModelview();
		update();
	} else {
		QPoint pos = modelviewI.map(event->pos());
		updateXY(pos.x(), pos.y());
	}
}

void Viewport::mousePressEvent(QMouseEvent *event)
{
	if (!active) {
		wasActive = false;
		emit activated(gradient);
		active = true;
	}

	if (event->button() == Qt::RightButton) {
		this->setCursor(Qt::ClosedHandCursor);
		lasty = event->y();
	}

	mouseMoveEvent(event);
}

void Viewport::mouseReleaseEvent(QMouseEvent * event)
{
	// in limiterMode, holdSelect is always set on first mouse action
	holdSelection = false;
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

	/* TODO: make sure that we use full space */

	updateModelview();
	update();
	event->accept();
}

void Viewport::keyPressEvent(QKeyEvent *event)
{
	bool dirty = false;

	switch (event->key()) {
	case Qt::Key_Plus:
		emit addSelection();
		break;
	case Qt::Key_Minus:
		emit remSelection();
		break;

	case Qt::Key_Up:
		if (!limiterMode && hover < nbins-1) {
			hover++;
			dirty = true;
		}
		break;
	case Qt::Key_Down:
		if (!limiterMode && hover > 0) {
			hover--;
			dirty = true;
		}
		break;
	case Qt::Key_Left:
		if (selection > 0) {
			selection--;
			emit bandSelected(selection, gradient);
			dirty = true;
		}
		break;
	case Qt::Key_Right:
		if (selection < dimensionality-1) {
			selection++;
			emit bandSelected(selection, gradient);
			dirty = true;
		}
		break;
	}

	if (dirty) {
		update();
		emit newOverlay();
	}
}

void Viewport::killHover()
{
	hover = -1;
	// make sure the drawing happens before next overlay cache update
	repaint();
}

bool Viewport::updateLimiter(int dim, int bin)
{
	std::pair<int, int> &l = limiters[dim];
	int &target = (std::abs(l.first - bin) < std::abs(l.second - bin)
		   ? l.first : l.second);
	if (target == bin)
		return false;

	target = bin;
	return true;
}
