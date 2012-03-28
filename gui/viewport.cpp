/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "viewport.h"
#include <stopwatch.h>
#include <iostream>
#include <cmath>
#include <QApplication>
#include <QMessageBox>
#include <QtCore>
#include <QPaintEvent>
#include <QRect>
#include <QPainter>

using namespace std;

Viewport::Viewport(QWidget *parent)
	: QGLWidget(QGLFormat(QGL::SampleBuffers), parent),
	  illuminant(NULL), selection(0), hover(-1), limiterMode(false),
	  active(false), wasActive(false), useralpha(1.f),
	  showLabeled(true), showUnlabeled(true), ignoreLabels(false),
	  overlayMode(false),
	  zoom(1.), shift(0), lasty(-1), holdSelection(false), activeLimiter(0),
	  cacheValid(false), clearView(false), implicitClearView(false),
	  drawMeans(true), drawRGB(false), drawHQ(true),
	  isHQ(false), shouldHQ(true),
	  yaxisWidth(0), vb(QGLBuffer::VertexBuffer)
{
	resizeTimer.setSingleShot(true);
	resizeTimer.setInterval(250);
	connect(&resizeTimer, SIGNAL(timeout()), this, SLOT(endNoHQ()));
}

Viewport::~Viewport()
{
}

void Viewport::reset(int bins, multi_img::Value bsize, multi_img::Value minv)
{
	nbins = bins;
	binsize = bsize;
	minval = minv;
	maxval = minv + (multi_img::Value)(bins - 1)*bsize;

	// reset hover value that would become inappropr.
	hover = -1;
	// reset limiters to most-lazy values
	setLimiters(0);

	// update y-axis (used by updateModelView())
	updateYAxis();

	// update coordinate system
	updateModelview();
}

void Viewport::updateYAxis()
{
	const int amount = 5;

	/* calculate raw numbers for y-axis */
	std::vector<float> ycoord(amount);
	float maximum = 0.f;
	for (int i = 0; i < amount; ++i) {
		float ifrac = (float)i*0.25*(float)(nbins - 1);
		ycoord[i] = maxval - ifrac*binsize;
		maximum = std::max(maximum, std::abs(ycoord[i]));
	}

	if (maximum == 0.f)
		return;

	/* find order of magnitude of maximum value */
	float roundAt = 0.001f; // we want 3 significant digits
	if (maximum >= 1.f) {
		do {
			maximum *= 0.1f;
			roundAt *= 10.f;
		} while (maximum >= 1.f);
	} else {
		while (maximum < 1.f) {
			maximum *= 10.f;
			roundAt *= 0.1f;
		};
	}

	/* set y-axis strings and find width of y-axis legend */
	yaxis.resize(amount);
	yaxisWidth = 0;
	QFontMetrics fm(font());
	for (int i = 0; i < amount; ++i) {
		float value = roundAt * std::floor(ycoord[i]/roundAt + 0.5f);
		yaxis[i] = QString().setNum(value, 'g', 3);
		yaxisWidth = std::max(yaxisWidth, fm.width(yaxis[i]));
	}
}

void Viewport::setLimiters(int label)
{
	if (label < 1) {	// not label
		limiters.assign(dimensionality, make_pair(0, nbins-1));
		if (label == -1) {	// use hover data
			int b = selection;
			int h = hover;
			limiters[b] = std::make_pair(h, h);
		}
	} else {                       // label holds data
		if (sets.size() > label && sets[label].totalweight > 0.f) {
			// use range from this label
			const std::vector<std::pair<int, int> > &b = sets[label].boundary;
			limiters.assign(b.begin(), b.end());
		} else
			setLimiters(0);
	}
}

void Viewport::prepareLines()
{
	// vole::Stopwatch watch("prepareLines");
	unsigned int total = shuffleIdx.size();

	makeCurrent();
	vb.setUsagePattern(QGLBuffer::StaticDraw);
	bool success = vb.create();
	if (!success) {
		QMessageBox::critical(this, "Drawing Error",
							  "Vertex Buffer Objects not supported."
							  "\nMake sure your graphics driver supports OpenGL 1.5 or later.");
		QApplication::quit();
	}
	success = vb.bind();
	if (!success) {
		QMessageBox::critical(this, "Drawing Error",
			"Drawing spectra cannot be continued. Please notify us about this"
			" problem, state error code 1 and what you did before it occured. Send an email to"
			" johannes.jordan@cs.fau.de. Thank you for your help!");
		return;
	}
	vb.allocate(total * dimensionality * sizeof(GLfloat) * 2);
	GLfloat *varr = (GLfloat*)vb.map(QGLBuffer::WriteOnly);
	if (!varr) {
		QMessageBox::critical(this, "Drawing Error",
			"Drawing spectra cannot be continued. Please notify us about this"
			" problem, state error code 2 and what you did before it occured. Send an email to"
			" johannes.jordan@cs.fau.de. Thank you for your help!");
		return;
	}

	int vidx = 0;
	for (unsigned int i = 0; i < total; ++i) {
		std::pair<int, QByteArray> &idx = shuffleIdx[i];
		BinSet &s = sets[idx.first];
		QByteArray &K = idx.second;
		Bin &b = s.bins[K];
		for (int d = 0; d < dimensionality; ++d) {
			qreal curpos;
			if (drawMeans) {
				curpos = (b.means[d] - minval)/binsize;
			} else {
				curpos = (unsigned char)K[d] + 0.5;
				if (illuminant_correction && illuminant)
					curpos *= (*illuminant)[d];
			}
			//b.points[d] = QPointF(d, curpos);
			varr[vidx++] = d;
			varr[vidx++] = curpos;
		}
	}
	vb.unmap();
	vb.release();
}

void Viewport::updateModelview()
{
	/* apply zoom and translation in window coordinates */
	qreal wwidth = width();
	qreal wheight = height()*zoom;
	int vshift = height()*shift;

	int hp = 20, vp = 12; // horizontal and vertical padding
	int vtp = 18; // lower padding for text (legend)
	int htp = yaxisWidth - 6; // left padding for text (legend)

	// if gradient, we discard one unit space intentionally for centering
	int d = dimensionality - (gradient? 0 : 1);
	qreal w = (wwidth  - 2*hp - htp)/(qreal)(d); // width of one unit
	qreal h = (wheight - 2*vp - vtp)/(qreal)(nbins - 1); // height of one unit
	int t = (gradient? w/2 : 0); // moving half a unit for centering

	modelview.reset();
	modelview.translate(hp + htp + t, vp + vshift);
	modelview.scale(w, -1*h); // -1 low values at bottom
	modelview.translate(0, -(nbins -1)); // shift for low values at bottom

	// set inverse
	modelviewI = modelview.inverted();
}

void Viewport::drawBins(QPainter &painter)
{
	// vole::Stopwatch watch("drawBins");
	painter.beginNativePainting();
	glEnable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glClear(GL_DEPTH_BUFFER_BIT);
	glDepthFunc(GL_LESS);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	bool success = vb.bind();
	if (!success) {
		QMessageBox::critical(this, "Drawing Error",
			"Drawing spectra cannot be continued. Please notify us about this"
			" problem, state error code 3 and what you did before it occured. Send an email to"
			" johannes.jordan@cs.fau.de. Thank you for your help!");
		painter.endNativePainting();
		return;
	}
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(2, GL_FLOAT, 0, 0);
	int currind = 0;

	/* check if we implicitely have a clear view */
	implicitClearView = (clearView || !active || (hover < 0 && !limiterMode));
	/* make sure that viewport shows "unlabeled" in the ignore label case */
	int start = ((showUnlabeled || ignoreLabels == 1) ? 0 : 1);
	int end = (showLabeled ? sets.size() : 1);

	for (unsigned int i = 0; i < shuffleIdx.size(); ++i) {
		std::pair<int, QByteArray> &idx = shuffleIdx[i];
		if (idx.first < start || idx.first >= end) {
			currind += dimensionality;
			continue;
		}

		BinSet &s = sets[idx.first];
		QByteArray &K = idx.second;
		Bin &b = s.bins[K];

		QColor &basecolor = s.label;
		QColor color = (drawRGB ? b.rgb : basecolor);
		qreal alpha;
		/* TODO: this is far from optimal yet. challenge is to give a good
		   view where important information is not lost, yet not clutter
		   the view with too much low-weight information */
		/* logarithm is used to prevent single data points to get lost.
		   this should be configurable. */
		if (i == 0)
			alpha = useralpha *
					(0.01 + 0.99*(log(b.weight+1) / log(s.totalweight)));
		else
			alpha = useralpha *
					(log(b.weight+1) / log(s.totalweight));
		color.setAlphaF(min(alpha, 1.));

		bool highlighted = false;
		if (!implicitClearView) {
			if (limiterMode) {
				highlighted = true;
				for (int i = 0; i < dimensionality; ++i) {
					unsigned char k = K[i];
					if (k < limiters[i].first || k > limiters[i].second)
						highlighted = false;
				}
			} else if ((unsigned char)K[selection] == hover)
				highlighted = true;
		}
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
		if (highlighted)
			glDepthMask(GL_TRUE); // write to depth mask -> stay in foreground
		else
			glDepthMask(GL_FALSE); // no writing -> may be overdrawn

		//painter.setPen(color);
		//painter.drawPolyline(b.points);
		qglColor(color);
		glDrawArrays(GL_LINE_STRIP, currind, dimensionality);
		currind += dimensionality;
	}
	vb.release();
		glDisable(GL_DEPTH_TEST);
		painter.endNativePainting();
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
			qreal y1 = limiters[i].first, y2 = limiters[i].second;
			if (illuminant) {
				y1 *= illuminant->at(selection);
				y2 *= illuminant->at(selection);
			}
			qreal h = nbins*0.01;
			if (h > y2 - y1)	// don't let them overlap, looks uncool
				h = y2 - y1;
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
	/// x-axis
	for (int i = 0; i < dimensionality; ++i) {
		QPointF l = modelview.map(QPointF(i - 1.f, 0.f));
		QPointF r = modelview.map(QPointF(i + 1.f, 0.f));
		QRectF rect(l, r);
		rect.setHeight(30.f);

		// only draw every xth label if we run out of space
		int stepping = std::max<int>(1, 120 / rect.width());
		if (i != selection) {
			if (i % stepping || std::abs(i - selection) < stepping)
				continue;
		}
		rect.adjust(-50.f, 0.f, 50.f, 0.f);

		if (i == selection)
			painter.setPen(Qt::red);
		painter.drawText(rect, Qt::AlignCenter, labels[i]);
		if (i == selection)	// revert back color
			painter.setPen(Qt::white);
	}

	/// y-axis
	for (size_t i = 0; i < yaxis.size(); ++i) {
		float ifrac = (float)(i)/(float)(yaxis.size()-1) * (float)(nbins - 1);
		QPointF b = modelview.map(QPointF(0.f, (float)(nbins - 1) - ifrac));
		b += QPointF(-8.f, 20.f); // draw left of data, vcenter alignment
		QPointF t = b;
		t -= QPointF(200.f, 40.f); // draw left of data, vcenter alignment
		QRectF rect(t, b);
		painter.drawText(rect, Qt::AlignVCenter | Qt::AlignRight, yaxis[i]);
	}
}

void Viewport::drawRegular()
{
	QPainter painter(this);
	QBrush background(QColor(15, 7, 15));
	painter.fillRect(rect(), background);

	isHQ = false;
/*	// draw without AA while user is messing around with mouse
	if (QApplication::mouseButtons() != Qt::NoButton && hasFocus())
		shouldHQ = false;*/
	if (drawHQ && shouldHQ) {
		painter.setRenderHint(QPainter::Antialiasing);
		isHQ = true;
	}

	drawLegend(painter);

	// needed in drawAxesBg
	painter.save();
	painter.setWorldTransform(modelview);
	drawAxesBg(painter);
	drawBins(painter);
	drawAxesFg(painter);
	// if you save and not restore, qt will lament about it
	painter.restore();
}

void Viewport::drawOverlay()
{
	QPainter painter(this);
	painter.drawImage(0, 0, cacheImg);
	painter.setRenderHint(QPainter::Antialiasing);

	QPolygonF poly = modelview.map(overlayPoints);
	QPen pen(Qt::black);
	pen.setWidth(5);
	painter.setPen(pen);
	painter.drawPolyline(poly);
	QPen pen2(Qt::yellow);
	pen2.setWidth(2);
	painter.setPen(pen2);
	painter.drawPolyline(poly);
}

void Viewport::activate()
{
	if (!active) {
		wasActive = false;
		emit activated();
		active = true;
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
	// quick drawing during resize
	startNoHQ();
	resizeTimer.start();

	updateModelview();
}

void Viewport::updateXY(int sel, int bin)
{
	bool emitOverlay = !wasActive;

	if (sel >= 0 && sel < dimensionality) {
		/// first handle sel -> band selection

		/* emit new band if either new selection or first input after
		   activating this view */
		if ((selection != sel && !holdSelection) || !wasActive) {
			wasActive = true;
			selection = sel;
			emitOverlay = true;
			emit bandSelected(sel, gradient);
		}

		// do this after the first chance to change selection (above)
		if (limiterMode)
			holdSelection = true;

		/// second handle bin -> intensity highlight
		if (illuminant && illuminant_correction)	/* correct y for illuminant */
			bin = std::floor(bin / illuminant->at(sel) + 0.5f);

		if (bin >= 0 && bin < nbins) {
			if (!limiterMode && (hover != bin)) {
				hover = bin;
				emitOverlay = true;
			}
			if (limiterMode && updateLimiter(selection, bin))
				emitOverlay = true;
		}
	}

	/// finally update
	if (emitOverlay) {
		update();	/* TODO: check for dual update! */
		emit newOverlay(selection);
	}
}

void Viewport::enterEvent(QEvent *)
{
	bool refresh = clearView;
	clearView = false;
	if (refresh)
		update();

/*	sloppy focus. debatable.
	if (active)
		return;
	emit bandSelected(selection, gradient);
	emit activated();
	active = true;
	update();
	emit newOverlay(-1);
*/
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
	activate(); // give ourselves active role if we don't have it yet
	startNoHQ();

	if (event->button() == Qt::RightButton) {
		this->setCursor(Qt::ClosedHandCursor);
		lasty = event->y();
	}

	mouseMoveEvent(event);
}

void Viewport::mouseReleaseEvent(QMouseEvent * event)
{
	// in limiterMode, holdSelect+act.Limiter is set on first mouse action
	holdSelection = false;
	activeLimiter = 0;

	if (event->button() == Qt::RightButton) {
		this->setCursor(Qt::ArrowCursor);
		lasty = -1;
	}

	endNoHQ();
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
	bool hoverdirt = false;

	switch (event->key()) {
	case Qt::Key_Plus:
		emit addSelection();
		break;
	case Qt::Key_Minus:
		emit remSelection();
		break;

	case Qt::Key_Up:
		if (!limiterMode && hover < nbins-2) {
			hover++;
			hoverdirt = true;
			dirty = true;
		}
		break;
	case Qt::Key_Down:
		if (!limiterMode && hover > 0) {
			hover--;
			hoverdirt = true;
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
	case Qt::Key_Space:
		drawHQ = !drawHQ;
		update();
		break;
	case Qt::Key_M:
		drawMeans = !drawMeans;
		prepareLines();
		update();
	}

	if (dirty) {
		update();
		emit newOverlay(hoverdirt ? selection : -1);
	}
}

void Viewport::killHover()
{
	clearView = true;

	if (!implicitClearView)
		// make sure the drawing happens before next overlay cache update
		repaint();
}

void Viewport::startNoHQ()
{
	shouldHQ = false;
}

void Viewport::endNoHQ()
{
	shouldHQ = true; // make sure we draw high quality again
	if (drawHQ && !isHQ)
		update();
}

bool Viewport::updateLimiter(int dim, int bin)
{
	std::pair<int, int> &l = limiters[dim];
	int *target;
	if (l.first == l.second) {
		target = (bin > l.first ? &l.second : &l.first);
	} else if (activeLimiter) {
		target = activeLimiter;
	} else {
		target = (std::abs(l.first-bin) < std::abs(l.second-bin) ?
				  &l.first : &l.second);
	}
	if (*target == bin)
		return false;

	*target = bin;
	activeLimiter = target;
	return true;
}
