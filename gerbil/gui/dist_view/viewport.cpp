/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "viewport.h"
#include "../iogui.h"
#include "../gerbil_gui_debug.h"

#include <qtopencv.h>
#include <stopwatch.h>

#include <iostream>
#include <cmath>
#include <QApplication>
#include <QMessageBox>
#include <QtCore>
#include <QGraphicsSceneEvent>
#include <QGraphicsProxyWidget>
#include <QGraphicsItem>
#include <QLayout>
#include <QRect>
#include <QPainter>
#include <boost/format.hpp>

using namespace std;

// default constants
const unsigned int Viewport::renderAtOnceStep 	 = 1024 * 1024 * 1024;
const unsigned int Viewport::spectrumRenderStep  = 10000;
const unsigned int Viewport::highlightRenderStep = 10000;

Viewport::Viewport(QGLWidget *target)
	: target(target), width(0), height(0),
	  ctx(new SharedData<ViewportCtx>(new ViewportCtx())),
	  sets(new SharedData<std::vector<BinSet> >(new std::vector<BinSet>())),
	  selection(0), hover(-1), limiterMode(false),
	  active(false), useralpha(1.f),
	  showLabeled(true), showUnlabeled(true),
	  overlayMode(false), highlightLabel(-1), illuminant_correction(false),
	  zoom(1.), shift(0), lasty(-1), holdSelection(false), activeLimiter(0),
	  clearView(false), implicitClearView(false),
	  drawMeans(true), drawRGB(false), drawHQ(true), drawingState(FOLDING),
	  yaxisWidth(0), vb(QGLBuffer::VertexBuffer),
	  fboSpectrum(NULL), fboHighlight(NULL), fboMultisamplingBlit(NULL),
	  controlItem(NULL)
{
	(*ctx)->wait = 1;
	(*ctx)->reset = 1;
	(*ctx)->ignoreLabels = false;

	initTimers();
}

Viewport::~Viewport()
{
	target->makeCurrent();
	delete fboSpectrum;
	delete fboHighlight;
	delete fboMultisamplingBlit;
}

/********* I N I T **************/

QGraphicsProxyWidget* Viewport::createControlProxy()
{
	// proxy for control widget
	controlItem = new QGraphicsProxyWidget();
	addItem(controlItem);

	return controlItem;
}

void Viewport::initTimers()
{
	resizeTimer.setSingleShot(true);
	connect(&resizeTimer, SIGNAL(timeout()), this, SLOT(resizeEpilog()));
	resizeTimer.start(0);

	scrollTimer.setSingleShot(true);
	connect(&scrollTimer, SIGNAL(timeout()), this, SLOT(updateTextures()));

	spectrumRenderedLines = 0;
	spectrumRenderTimer.setSingleShot(true);
	connect(&spectrumRenderTimer, SIGNAL(timeout()), this, SLOT(continueDrawingSpectrum()));

	highlightRenderedLines = 0;
	highlightRenderTimer.setSingleShot(true);
	connect(&highlightRenderTimer, SIGNAL(timeout()), this, SLOT(continueDrawingHighlight()));
}

/********* S T A T E ********/

void Viewport::activate()
{
	if (!active) {
		active = true;
		{	// yeah, maybe really add the member just for this one occasion..
			SharedDataLock ctxlock(ctx->mutex);
			emit activated((*ctx)->type);
		}
		emit bandSelected(selection);

		// this is a sometimes redundant operation (matrix maybe still valid)
		if (limiterMode)
			emit requestOverlay(limiters, -1);
		else
			emit requestOverlay(selection, hover);
	}
}

void Viewport::removePixelOverlay()
{
	overlayMode = false;
	if (target->isVisible()) // TODO: does this work?
		update();
}

void Viewport::insertPixelOverlay(const QPolygonF &points)
{
	overlayPoints = points;
	overlayMode = true;
	update();
}

void Viewport::changeIlluminant(cv::Mat1f illum)
{
	illuminant = illum;

	if (!illuminant_correction) { // vertices need to change
		// TODO: should this not call prepareLines() also?
		updateTextures();
	} else {
		update();
	}
}

void Viewport::setIlluminantIsApplied(bool applied)
{
	// only set it to true, never to false again (you cannot unapply)
	if (applied) {
		illuminant_correction = true;
	}
}


void Viewport::setIlluminationCurveShown(bool show)
{
	if (show) {
		// HACK
		// nothing, illuminant is always drawn if coefficients viewport->illuminant
		// non-empty. Should really be a bool in viewport.
	} else {
		cv::Mat1f empty;
		changeIlluminant(empty);
	}
}

void Viewport::reset()
{
	// reset values that would become inappropr.
	selection = 0;
	hover = -1;

	// reset limiters to most-lazy values
	setLimiters(0);

	// update y-axis (used by updateModelView())
	updateYAxis();

	// update coordinate system
	updateModelview();
}

void Viewport::setLimiters(int label)
{
	if (label < 1) {	// not label
		SharedDataLock ctxlock(ctx->mutex);
		limiters.assign((*ctx)->dimensionality, make_pair(0, (*ctx)->nbins-1));
		if (label == -1) {	// use hover data
			int b = selection;
			int h = hover;
			limiters[b] = std::make_pair(h, h);
		}
	} else {            // label holds data
		SharedDataLock setslock(sets->mutex);
		if ((int)(*sets)->size() > label && (**sets)[label].totalweight > 0) {
			// use range from this label
			const std::vector<std::pair<int, int> > &b = (**sets)[label].boundary;
			limiters.assign(b.begin(), b.end());
		} else {
			setLimiters(0);
		}
	}
}

void Viewport::rebuild()
{
	// guess: we want the lock to carry over both methods..
	SharedDataLock ctxlock(ctx->mutex);
	SharedDataLock setslock(sets->mutex);

	// TODO: is this the right place?
	if (selection > (*ctx)->dimensionality)
		selection = 0;

	// make sure band view is on track
	if (active) {
		emit bandSelected(selection);
	}

	prepareLines();
	updateTextures();
}

void Viewport::killHover()
{
	clearView = true;

	if (!implicitClearView)
		// make sure the drawing happens before next overlay cache update
		// TODO: still relevant?
		target->repaint();
}

void Viewport::highlightSingleLabel(int index)
{
	highlightLabel = index;
	updateTextures();
}

void Viewport::setAlpha(float alpha)
{
	useralpha = alpha;
	updateTextures(Viewport::RM_STEP, Viewport::RM_SKIP);
}

void Viewport::setLimitersMode(bool enabled)
{
	limiterMode = enabled;
	updateTextures(Viewport::RM_SKIP, Viewport::RM_STEP);
	if (!active) {
		activate(); // will call requestOverlay()
	} else {
		// this is a sometimes redundant operation (matrix maybe still valid)
		if (limiterMode)
			emit requestOverlay(limiters, -1);
		else
			emit requestOverlay(selection, hover);
	}
}

void Viewport::startNoHQ(bool resize)
{
	if (resize)
		drawingState = RESIZE;
	else
		drawingState = (drawingState == HIGH_QUALITY ? HIGH_QUALITY_QUICK : QUICK);
}

bool Viewport::endNoHQ()
{
	bool dirty = true;
	if (drawingState == HIGH_QUALITY || drawingState == HIGH_QUALITY_QUICK)
		dirty = false;
	if (!drawHQ && drawingState == QUICK)
		dirty = false;

	drawingState = (drawHQ ? HIGH_QUALITY : QUICK);
	return dirty;
}

bool Viewport::updateLimiter(int dim, int bin)
{
	std::pair<int, int> &l = limiters[dim];
	int *target;
	if (l.first == l.second) { // both top and bottom are same
		target = (bin > l.first ? &l.second : &l.first);
	} else if (activeLimiter) {
		target = activeLimiter;
	} else { // choose closest between top and bottom
		target = (std::abs(l.first-bin) < std::abs(l.second-bin) ?
				  &l.first : &l.second);
	}
	if (*target == bin) // no change
		return false;

	*target = bin;
	activeLimiter = target;
	return true;
}

void Viewport::screenshot()
{
	drawingState = SCREENSHOT;
	updateTextures(RM_FULL, RM_FULL);
	target->repaint();
	cacheImg = target->grabFrameBuffer(); // TODO: more elegant with fbo/qgv

	// write out
	cv::Mat output = vole::QImage2Mat(cacheImg);
	IOGui io("Screenshot File", "screenshot", target);
	io.writeFile(QString(), output);

	// reset drawingState and draw again nicely
	if (endNoHQ())
		updateTextures();
}

/***** D R A W I N G ******/

void Viewport::updateYAxis()
{
	const int amount = 5;

	/* calculate raw numbers for y-axis */
	std::vector<float> ycoord(amount);
	float maximum = 0.f;
	for (int i = 0; i < amount; ++i) {
		SharedDataLock ctxlock(ctx->mutex);
		float ifrac = (float)i*0.25*(float)((*ctx)->nbins - 1);
		ycoord[i] = (*ctx)->maxval - ifrac * (*ctx)->binsize;
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

void Viewport::updateModelview()
{
	SharedDataLock ctxlock(ctx->mutex);

	/* apply zoom and translation in window coordinates */
	qreal wwidth = width;
	qreal wheight = height*zoom;
	int vshift = height*shift;

	int hp = 20, vp = 12; // horizontal and vertical padding
	int vtp = 18; // lower padding for text (legend)
	int htp = yaxisWidth - 6; // left padding for text (legend)

	// if gradient, we discard one unit space intentionally for centering
	int d = (*ctx)->dimensionality - ((*ctx)->type == representation::GRAD ? 0 : 1);
	qreal w = (wwidth  - 2*hp - htp)/(qreal)(d); // width of one unit
	qreal h = (wheight - 2*vp - vtp)/(qreal)((*ctx)->nbins); // height of one unit
	int t = ((*ctx)->type == representation::GRAD ? w/2 : 0); // moving half a unit for centering

	modelview.reset();
	modelview.translate(hp + htp + t, vp + vshift);
	modelview.scale(w, -1*h); // -1 low values at bottom
	modelview.translate(0, -((*ctx)->nbins)); // shift for low values at bottom

	// set inverse
	modelviewI = modelview.inverted();
}

void Viewport::prepareLines()
{
	// lock context and sets
	SharedDataLock ctxlock(ctx->mutex);
	SharedDataLock setslock(sets->mutex);
	(*ctx)->wait.fetch_and_store(0); // set to zero: our data will be usable
	if ((*ctx)->reset.fetch_and_store(0)) // is true if it was 1 before
		reset();

	// first step (cpu only)
	Compute::preparePolylines(**ctx, **sets, shuffleIdx);

	// second step (cpu -> gpu)
	target->makeCurrent();
	int success = Compute::storeVertices(**ctx, **sets, shuffleIdx, vb,
										 drawMeans, illuminant_correction,
										 illuminant);

	// gracefully fail if there is a problem with VBO support
	switch (success) {
	case 0:
		return;
	case -1:
		QMessageBox::critical(target, "Drawing Error",
			"Vertex Buffer Objects not supported.\n"
			"Make sure your graphics driver supports OpenGL 1.5 or later.");
		QApplication::quit();
		exit(1);
	default:
		QMessageBox::critical(target, "Drawing Error",
			QString("Drawing spectra cannot be continued. "
					"Please notify us about this problem, state error code %1 "
					"and what actions led up to this error. Send an email to"
			" johannes.jordan@cs.fau.de. Thank you for your help!").arg(success));
		return;
	}
}

void Viewport::drawBins(QPainter &painter, QTimer &renderTimer,
	unsigned int &renderedLines, unsigned int renderStep, bool onlyHighlight)
{
	SharedDataLock ctxlock(ctx->mutex);
	SharedDataLock setslock(sets->mutex); // TODO: this also locks shuffleIdx implic.

	// vole::Stopwatch watch("drawBins");
	painter.beginNativePainting();
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	bool success = vb.bind();
	if (!success) {
		QMessageBox::critical(target, "Drawing Error",
			"Drawing spectra cannot be continued. "
			"Please notify us about this problem, state error code 3 "
			"and what actions led up to this error. Send an email to"
			" johannes.jordan@cs.fau.de. Thank you for your help!");
		painter.endNativePainting();
		return;
	}
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(2, GL_FLOAT, 0, 0);
	int currind = renderedLines * (*ctx)->dimensionality;

	/* check if we implicitely have a clear view */
	implicitClearView = (clearView || !active || drawingState == SCREENSHOT ||
						 (hover < 0 && !limiterMode));
	/* make sure that viewport shows "unlabeled" in the ignore label case */
	int start = ((showUnlabeled || (*ctx)->ignoreLabels == 1) ? 0 : 1);
	int end = (showLabeled ? (*sets)->size() : 1);
	int single = ((*ctx)->ignoreLabels ? -1 : highlightLabel);

	unsigned int total = shuffleIdx.size();
	unsigned int first = renderedLines;
	unsigned int last = (renderedLines + renderStep) < total ? (renderedLines + renderStep) : total;
	for (unsigned int i = first; i < last; ++i) {
		std::pair<int, BinSet::HashKey> &idx = shuffleIdx[i];
		if ((idx.first < start || idx.first >= end) && !(idx.first == single)) {
			currind += (*ctx)->dimensionality;
			if (last < total)
				++last; // increase loop count
			continue;
		}
		
		BinSet::HashKey &K = idx.second;

		bool highlighted = false;
		if (!implicitClearView && onlyHighlight) {
			if (limiterMode) {
				highlighted = true;
				for (int i = 0; i < (*ctx)->dimensionality; ++i) {
					unsigned char k = K[i];
					if (k < limiters[i].first || k > limiters[i].second)
						highlighted = false;
				}
			} else if ((unsigned char)K[selection] == hover)
				highlighted = true;
		}

		if (!highlighted && onlyHighlight) {
			currind += (*ctx)->dimensionality;
			if (last < total)
				++last; // increase loop count
			continue;
		}

		BinSet &s = (**sets)[idx.first];
		Bin &b = s.bins.equal_range(K).first->second;

		QColor color = determineColor((drawRGB ? b.rgb : s.label),
									  b.weight, s.totalweight,
									  highlighted && onlyHighlight,
									  idx.first == single);
		target->qglColor(color);
		glDrawArrays(GL_LINE_STRIP, currind, (*ctx)->dimensionality);
		currind += (*ctx)->dimensionality;
	}
	vb.release();
	painter.endNativePainting();

	renderedLines += (last - first);
	if (renderedLines < total) {
		if (renderedLines <= renderStep) {
			renderTimer.start(150);
		} else {
			renderTimer.start(0);
		}
	}
}

QColor Viewport::determineColor(const QColor &basecolor,
								float weight, float totalweight,
								bool highlighted, bool single)
{
	QColor color = basecolor;
	qreal alpha;
	/* TODO: this is far from optimal yet. challenge is to give a good
	   view where important information is not lost, yet not clutter
	   the view with too much low-weight information */
	/* logarithm is used to prevent single data points to get lost.
	   this should be configurable. */
	alpha = useralpha *
			(0.01 + 0.99*(log(weight+1) / log(totalweight)));
	//	TODO: option
	//      (0.01 + 0.99*(weight / totalweight));
	color.setAlphaF(min(alpha, 1.)); // cap at 1

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

	// recolor singleLabel
	if (!highlighted && implicitClearView && single) {
		color.setRgbF(1., 1., 0., color.alphaF());
	}
	return color;
}

void Viewport::drawAxesFg(QPainter *painter)
{

	SharedDataLock ctxlock(ctx->mutex);

	if (drawingState == SCREENSHOT)
		return;
	if (selection < 0 || selection >= (*ctx)->dimensionality)
		return;

	// draw selection in foreground
	if (active)
		painter->setPen(Qt::red);
	else
		painter->setPen(Qt::gray);
	qreal top = ((*ctx)->nbins);
	if (!illuminant.empty())
		top *= illuminant.at(selection);
	painter->drawLine(QPointF(selection, 0.), QPointF(selection, top));

	// draw limiters
	if (limiterMode) {
		painter->setPen(Qt::red);
		for (int i = 0; i < (*ctx)->dimensionality; ++i) {
			qreal y1 = limiters[i].first, y2 = limiters[i].second + 1;
			if (!illuminant.empty()) {
				y1 *= illuminant.at(selection);
				y2 *= illuminant.at(selection);
			}
			qreal h = (*ctx)->nbins*0.01;
			if (h > y2 - y1)	// don't let them overlap, looks uncool
				h = y2 - y1;
			QPolygonF polygon;
			polygon << QPointF(i - 0.25, y1 + h)
					<< QPointF(i - 0.25, y1)
					<< QPointF(i + 0.25, y1)
					<< QPointF(i + 0.25, y1 + h);
			painter->drawPolyline(polygon);
			polygon.clear();
			polygon << QPointF(i - 0.25, y2 - h)
					<< QPointF(i - 0.25, y2)
					<< QPointF(i + 0.25, y2)
					<< QPointF(i + 0.25, y2 - h);
			painter->drawPolyline(polygon);
		}
	}
}
void Viewport::drawAxesBg(QPainter *painter)
{
	SharedDataLock ctxlock(ctx->mutex);

	// draw axes in background
	painter->setPen(QColor(64, 64, 64));
	QPolygonF poly;
	if (!illuminant.empty()) {
		for (int i = 0; i < (*ctx)->dimensionality; ++i) {
			qreal top = ((*ctx)->nbins-1) * illuminant.at(i);
			painter->drawLine(QPointF(i, 0.), QPointF(i, top));
			poly << QPointF(i, top);
		}
		poly << QPointF((*ctx)->dimensionality-1, (*ctx)->nbins-1);
		poly << QPointF(0, (*ctx)->nbins-1);
	} else {
		for (int i = 0; i < (*ctx)->dimensionality; ++i)
			painter->drawLine(i, 0, i, (*ctx)->nbins);
	}

	// visualize illuminant
	if (!illuminant.empty()) {
		QPolygonF poly2 = modelview.map(poly);
		poly2.translate(0., -5.);
		painter->restore();
		QBrush brush(QColor(32, 32, 32), Qt::Dense3Pattern);
		painter->setBrush(brush);
		painter->setPen(Qt::NoPen);
		painter->drawPolygon(poly2);
		painter->setPen(Qt::white);
		poly2.remove((*ctx)->dimensionality, 2);
		painter->drawPolyline(poly2);
		painter->save();
		painter->setWorldTransform(modelview);
	}
}

void Viewport::drawLegend(QPainter *painter)
{
	SharedDataLock ctxlock(ctx->mutex);

	assert((*ctx)->labels.size() == (unsigned int)(*ctx)->dimensionality);

	painter->setPen(Qt::white);
	/// x-axis
	for (int i = 0; i < (*ctx)->dimensionality; ++i) {
		QPointF l = modelview.map(QPointF(i - 1.f, 0.f));
		QPointF r = modelview.map(QPointF(i + 1.f, 0.f));
		QRectF rect(l, r);
		rect.setHeight(30.f);

		// only draw every xth label if we run out of space
		int stepping = std::max<int>(1, 150 / rect.width());

		if (drawingState != SCREENSHOT) {
			// select to draw selection and not its surroundings
			if (i != selection) {
				if (i % stepping || std::abs(i - selection) < stepping)
					continue;
			}
		} else {
			// select to draw only regular axis text
			if (i % stepping)
				continue;
		}
		rect.adjust(-50.f, 0.f, 50.f, 0.f);

		bool highlight = (i == selection && drawingState != SCREENSHOT);
		if (highlight)
			painter->setPen(Qt::red);
		painter->drawText(rect, Qt::AlignCenter, (*ctx)->labels[i]);
		if (highlight)	// revert back color
			painter->setPen(Qt::white);
	}

	/// y-axis
	for (size_t i = 0; i < yaxis.size(); ++i) {
		float ifrac = (float)(i)/(float)(yaxis.size()-1) * (float)((*ctx)->nbins - 1);
		QPointF b = modelview.map(QPointF(0.f, (float)((*ctx)->nbins - 1) - ifrac));
		b += QPointF(-8.f, 20.f); // draw left of data, vcenter alignment
		QPointF t = b;
		t -= QPointF(200.f, 40.f); // draw left of data, vcenter alignment
		QRectF rect(t, b);
		painter->drawText(rect, Qt::AlignVCenter | Qt::AlignRight, yaxis[i]);
	}
}

void Viewport::drawOverlay(QPainter *painter)
{
	painter->save();
	QPolygonF poly = modelview.map(overlayPoints);
	QPen pen(Qt::black);
	pen.setWidth(5);
	painter->setPen(pen);
	painter->drawPolyline(poly);
	QPen pen2(Qt::yellow);
	pen2.setWidth(2);
	painter->setPen(pen2);
	painter->drawPolyline(poly);
	painter->restore();
}

void Viewport::drawWaitMessage(QPainter *painter)
{
	QRect rect(0, 0, width, height);
	painter->save();
	// darken
	painter->fillRect(rect, QColor(0, 0, 0, 127));

	// text in larger size with nice color
	painter->setPen(QColor(255, 230, 0));
	QFont tmp(font());
	tmp.setPointSize(tmp.pointSize() * 1.75);
	painter->setFont(tmp);
	painter->drawText(rect, Qt::AlignCenter,
					 QString::fromUtf8("Calculating…"));
	painter->restore();
}

void Viewport::continueDrawingSpectrum()
{
	if (!fboSpectrum)
		return;

	SharedDataLock ctxlock(ctx->mutex);
	SharedDataLock setslock(sets->mutex);

	if ((*sets)->empty() || (*ctx)->wait || drawingState == FOLDING)
		return;

	setslock.unlock();
	ctxlock.unlock();

	QPainter painter(fboSpectrum);

	if (drawingState == HIGH_QUALITY || drawingState == SCREENSHOT)
		painter.setRenderHint(QPainter::Antialiasing);

	painter.save();
	painter.setWorldTransform(modelview);
	drawBins(painter, spectrumRenderTimer,  spectrumRenderedLines, spectrumRenderStep, false);
	painter.restore();

	update();
}

void Viewport::continueDrawingHighlight()
{
	if (!fboHighlight)
		return;

	SharedDataLock ctxlock(ctx->mutex);
	SharedDataLock setslock(sets->mutex);

	if ((*sets)->empty() || (*ctx)->wait || drawingState == FOLDING)
		return;

	setslock.unlock();
	ctxlock.unlock();

	QPainter painter(fboHighlight);

	if (drawingState == HIGH_QUALITY || drawingState == SCREENSHOT)
		painter.setRenderHint(QPainter::Antialiasing);

	painter.save();
	painter.setWorldTransform(modelview);
	drawBins(painter, highlightRenderTimer,  highlightRenderedLines, highlightRenderStep, true);
	painter.restore();

	update();
}

void Viewport::updateTextures(RenderMode spectrum, RenderMode highlight)
{
	if (!fboSpectrum || !fboHighlight)
		return;

	if (spectrum) {
		spectrumRenderTimer.stop();
		spectrumRenderedLines = 0;
	}

	if (highlight) {
		highlightRenderTimer.stop();
		highlightRenderedLines = 0;
	}

	SharedDataLock ctxlock(ctx->mutex);
	SharedDataLock setslock(sets->mutex);

	QPainter spectrumPainter(fboSpectrum);
	QPainter highlightPainter(fboHighlight);

	//GGDBGM("before spectrum\n");
	//GGDBGM(boost::format("spectrum=%1%, fboSpectrum=%2%\n")
	//	   % (int)spectrum % fboSpectrum );
	QRect rect(0, 0, width, height);
	if (spectrum != RM_SKIP) {
		spectrumPainter.setCompositionMode(QPainter::CompositionMode_Source);
		spectrumPainter.fillRect(rect, Qt::transparent);
		spectrumPainter.setCompositionMode(QPainter::CompositionMode_SourceOver);
	}
	//GGDBGM("after spectrum\n");

	if (highlight != RM_SKIP) {
		highlightPainter.setCompositionMode(QPainter::CompositionMode_Source);
		highlightPainter.fillRect(rect, Qt::transparent);
		highlightPainter.setCompositionMode(QPainter::CompositionMode_SourceOver);
	}

	if ((*sets)->empty() || (*ctx)->wait || drawingState == FOLDING)
		return;

	if (drawingState == HIGH_QUALITY || drawingState == SCREENSHOT) {
		if (spectrum)
			spectrumPainter.setRenderHint(QPainter::Antialiasing);
		if (highlight)
			highlightPainter.setRenderHint(QPainter::Antialiasing);
	} else {
		// even if we had HQ last time, this time it will be dirty!
		if (drawingState == HIGH_QUALITY_QUICK)
			drawingState = QUICK;
	}

	if (spectrum) {
		spectrumPainter.save();
		spectrumPainter.setWorldTransform(modelview);
		drawBins(spectrumPainter, spectrumRenderTimer,  spectrumRenderedLines, 
			(spectrum == RM_FULL) ? renderAtOnceStep : spectrumRenderStep, false);
		spectrumPainter.restore();
	}

	if (highlight) {
		highlightPainter.save();
		highlightPainter.setWorldTransform(modelview);
		drawBins(highlightPainter, highlightRenderTimer, highlightRenderedLines,
			(highlight == RM_FULL) ? renderAtOnceStep : highlightRenderStep, true);
		highlightPainter.restore();
	}

	update();
}

void Viewport::toggleLabeled(bool enabled)
{
	showLabeled = enabled;
	updateTextures();
}

void Viewport::toggleUnlabeled(bool enabled)
{
	showUnlabeled = enabled;
	updateTextures();
}

void Viewport::drawBackground(QPainter *painter, const QRectF &rectx)
{
	SharedDataLock ctxlock(ctx->mutex);
	SharedDataLock setslock(sets->mutex);

	// TODO: maybe not needed here
	target->makeCurrent();
	int nwidth = painter->device()->width();
	int nheight = painter->device()->height();
	if (nwidth != width || nheight != height) {
		width = nwidth;
		height = nheight;
		resizeScene();
	}

	painter->setRenderHint(QPainter::Antialiasing);

	QRect rect(0, 0, width, height);
	if (drawingState == SCREENSHOT) {
		painter->fillRect(rect, Qt::black);
	} else {
		painter->fillRect(rect, QColor(15, 7, 15));
	}

	if (!(*sets)->empty() && !(*ctx)->wait) {
		drawLegend(painter);
		painter->save();
		painter->setWorldTransform(modelview);
		drawAxesBg(painter);
		painter->restore();
	}

	if (fboSpectrum) {
		// TODO: this is only for multisample FBO, which we currently do NOT use
		QGLFramebufferObject::blitFramebuffer(fboMultisamplingBlit,
											  rect, fboSpectrum, rect);
		target->drawTexture(rect, fboMultisamplingBlit->texture());
	}

	implicitClearView = (clearView || !active || drawingState == SCREENSHOT || (hover < 0 && !limiterMode));
	
	if (fboHighlight && !overlayMode && !implicitClearView) {
		QGLFramebufferObject::blitFramebuffer(fboMultisamplingBlit, rect,
											  fboHighlight, rect);
		target->drawTexture(rect, fboMultisamplingBlit->texture());
	}

	if (!(*sets)->empty() && !(*ctx)->wait) {
		painter->save();
		painter->setWorldTransform(modelview);
		drawAxesFg(painter);
		painter->restore();
	}

	if (overlayMode) {
		drawOverlay(painter);
	}

	//GGDBGM(boost::format("%1%  (*sets)->empty()=%2% || (*ctx)->wait=%3% || disabled=%4%   this=%5%\n")
	//			 % (*ctx)->type %(*sets)->empty() %(*ctx)->wait %(!isEnabled()) %this);

	if ((*sets)->empty() || (*ctx)->wait) {
		drawWaitMessage(painter);
	}
}

void Viewport::resizeScene()
{
	/* resize framebuffers */
	target->makeCurrent();

	QGLFramebufferObjectFormat format;
	format.setAttachment(QGLFramebufferObject::NoAttachment);
	format.setSamples(1); // TODO: configurable? test with other GPUs.

	// use floating point for better alpha accuracy!
	// TODO: We don't need this for highlights. do we care?
	format.setInternalTextureFormat(GL_RGBA16F); // TODO RGBA32F yet looks better!

	delete fboSpectrum;
	delete fboHighlight;
	delete fboMultisamplingBlit;

	fboSpectrum = new QGLFramebufferObject(width, height, format);
	fboHighlight = new QGLFramebufferObject(width, height, format);
	fboMultisamplingBlit = new QGLFramebufferObject(width, height);

	if (drawingState != FOLDING) {
		// quick drawing during resize
		startNoHQ(true);
		resizeTimer.start(150);
	}

	/* update transformation matrix */
	updateModelview();

	/* update control widget */
	controlItem->setMinimumHeight(height);
}

void Viewport::resizeEpilog()
{
	if (endNoHQ())
		updateTextures();
}

/********** M O U S E / K E Y B .    I N P U T  ***********/

bool Viewport::updateXY(int sel, int bin)
{
	SharedDataLock ctxlock(ctx->mutex);

	if (sel < 0 || sel >= (*ctx)->dimensionality)
		return false;

	bool highlightChanged = false;

	/* first: handle sel -> band selection */

	if (selection != sel && !holdSelection) {
		selection = sel;
		// selection results in new band to be shown
		emit bandSelected(selection);
		// it also results in a new overlay
		emit requestOverlay(selection, hover);
		// and we changed our own highlight
		highlightChanged = true;
	}

	// do this after the first chance to change selection (above)
	if (limiterMode)
		// no accidential jumping to limiters of other bands
		holdSelection = true;

	/* second: handle bin -> intensity highlight */

	/* correct y for illuminant */
	if (!illuminant.empty() && illuminant_correction)
		bin = std::floor(bin / illuminant.at(sel) + 0.5f);

	if (bin >= 0 && bin < (*ctx)->nbins) {
		if (!limiterMode && (hover != bin)) {
			hover = bin;
			emit requestOverlay(selection, hover);
			highlightChanged = true;
		}
		if (limiterMode && updateLimiter(selection, bin)) {
			emit requestOverlay(limiters, selection);
			highlightChanged = true;
		}
	}

	return highlightChanged;
}

void Viewport::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	// check for scene elements first (we are technically the background)
	QGraphicsScene::mouseMoveEvent(event);
	if (event->isAccepted())
		return;

	bool needUpdate = false, needTextureUpdate = false;

	/* we could have entered the window right now, need to refresh
	   in case there was a pixel highlight (just a guess!!) */
	needUpdate = clearView;
	clearView = false;

	if (event->buttons() & Qt::LeftButton) {
		/* cursor control */

		QPointF pos = modelviewI.map(event->scenePos());
		// add .5 to x for rounding
		needTextureUpdate = updateXY(pos.x() + 0.5f, pos.y());

	} else if (event->buttons() & Qt::RightButton) {
		/* panning movement */

		if (lasty < 0)
			return;

		shift += (event->scenePos().y() - lasty)/(qreal)height;
		lasty = event->scenePos().y(); // TODO: will be done by qgraphicsscene!

		/* TODO: make sure that we use full visible space */

		updateModelview();
		spectrumRenderTimer.stop();
		highlightRenderTimer.stop();
		scrollTimer.start(10);

	} else {
		/* access to control widget */
		if (event->scenePos().x() < 20) {
			emit scrollInControl();
		} else if (event->scenePos().x() >
				 controlItem->boundingRect().right() + 10) {
			emit scrollOutControl();
		}
	}

	if (needTextureUpdate) {
		// calls update()
		updateTextures(RM_SKIP, limiterMode ? RM_STEP : RM_FULL);
	} else if (needUpdate) {
		update();
	}

	event->accept();
}

void Viewport::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	activate(); // give ourselves active role if we don't have it yet

	// check for scene elements first (we are technically the background)
	QGraphicsScene::mousePressEvent(event);
	if (event->isAccepted())
		return;

	startNoHQ();

	if (event->button() == Qt::RightButton) {
		target->setCursor(Qt::ClosedHandCursor);
		lasty = event->scenePos().y(); // TODO: qgraphicsscene
	}

	mouseMoveEvent(event);
}

void Viewport::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	// check for scene elements first (we are technically the background)
	QGraphicsScene::mouseReleaseEvent(event);
	if (event->isAccepted())
		return;

	/* in limiterMode, holdSelect+act.Limiter is set on first mouse action,
	 * so we reset them now as mouse actions are finished */
	holdSelection = false;
	activeLimiter = 0;

	if (event->button() == Qt::RightButton) {
		target->setCursor(Qt::ArrowCursor);
		lasty = -1;
	}

	if (endNoHQ()) {
		updateTextures((event->button() == Qt::RightButton)
					   ? RM_STEP : RM_SKIP, RM_STEP);
	}

	event->accept();
}

void Viewport::wheelEvent(QGraphicsSceneWheelEvent *event)
{
	// check for scene elements first (we are technically the background)
	QGraphicsScene::wheelEvent(event);
	if (event->isAccepted())
		 return;

	// zoom in or out
	qreal oldzoom = zoom;
	if (event->delta() > 0)
		zoom *= 1.25;
	else
		zoom = max(zoom * 0.80, 1.);

	// adjust shift to new zoom
	shift += ((oldzoom - zoom) * 0.5);

	/* TODO: make sure that we use full space */

	updateModelview();
	updateTextures();
	event->accept();
}

void Viewport::keyPressEvent(QKeyEvent *event)
{
	bool highlightAltered = false;

	switch (event->key()) {
	case Qt::Key_S:
		screenshot();
		break;

	case Qt::Key_Plus:
		emit addSelectionRequested();
		break;
	case Qt::Key_Minus:
		emit remSelectionRequested();
		break;

	case Qt::Key_Up:
		{
			SharedDataLock ctxlock(ctx->mutex);
			if (!limiterMode && hover < (*ctx)->nbins-2) {
				hover++;
				requestOverlay(selection, hover);
				highlightAltered = true;
			}
		}
		break;
	case Qt::Key_Down:
		if (!limiterMode && hover > 0) {
			hover--;
			requestOverlay(selection, hover);
			highlightAltered = true;
		}
		break;
	case Qt::Key_Left:
		{
			SharedDataLock ctxlock(ctx->mutex);
			if (selection > 0) {
				selection--;
				emit bandSelected(selection);
				if (!limiterMode) // we do not touch the limiters
					emit requestOverlay(selection, hover);
				highlightAltered = true;
			}
		}
		break;
	case Qt::Key_Right:
		{
			SharedDataLock ctxlock(ctx->mutex);
			if (selection < (*ctx)->dimensionality-1) {
				selection++;
				emit bandSelected(selection);
				if (!limiterMode) // we do not touch the limiters
					emit requestOverlay(selection, hover);
				highlightAltered = true;
			}
		}
		break;

	case Qt::Key_Space:
		drawHQ = !drawHQ;
		if (drawHQ) {
			if (endNoHQ())
				updateTextures();
		} else {
			startNoHQ();
		}
		updateTextures();
		break;
	case Qt::Key_M:
		drawMeans = !drawMeans;
		rebuild();
	}

	if (highlightAltered) {
		updateTextures(RM_SKIP, RM_FULL);
	}
	event->accept();
}

bool Viewport::event(QEvent *event)
{
	// we only deal with leaveEvent
	if (event->type() == QEvent::Leave) {
		emit scrollOutControl();
		return true;
	}

	return QGraphicsScene::event(event);
}
