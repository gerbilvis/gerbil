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
#include <QApplication>
#include <QMessageBox>
#include <QGraphicsProxyWidget>
#include <QGraphicsItem>
#include <QPainter>
#include <QSignalMapper>
#include <boost/format.hpp>

Viewport::Viewport(representation::t type, QGLWidget *target)
	: type(type), target(target), width(0), height(0),
	  ctx(new SharedData<ViewportCtx>(new ViewportCtx())),
	  sets(new SharedData<std::vector<BinSet> >(new std::vector<BinSet>())),
	  selection(0), hover(-1), limiterMode(false),
	  active(false), useralpha(1.f),
	  showLabeled(true), showUnlabeled(true),
	  overlayMode(false), highlightLabel(-1),
	  illuminant_show(true),
	  zoom(1.), shift(0), lasty(-1), holdSelection(false), activeLimiter(0),
	  drawLog(true), drawMeans(true), drawRGB(false), drawHQ(true),
	  bufferFormat(RGBA16F),
	  drawingState(HIGH_QUALITY), yaxisWidth(0), vb(QGLBuffer::VertexBuffer)
{
	(*ctx)->wait = 1;
	(*ctx)->reset = 1;
	(*ctx)->ignoreLabels = false;

	initTimers();
}

Viewport::~Viewport()
{
	target->makeCurrent();
	for (int i = 0; i < 2; ++i) {
		delete buffers[i].fbo;
		delete buffers[i].blit;
	}
}

/********* I N I T **************/

void Viewport::initTimers()
{
	resizeTimer.setSingleShot(true);
	connect(&resizeTimer, SIGNAL(timeout()), this, SLOT(resizeScene()));

	scrollTimer.setSingleShot(true);
	connect(&scrollTimer, SIGNAL(timeout()), this, SLOT(updateBuffers()));

	QSignalMapper *mapper = new QSignalMapper();
	for (int i = 0; i < 2; ++i) {
		buffers[i].renderedLines = 0;
		buffers[i].renderTimer.setSingleShot(true);
		connect(&buffers[i].renderTimer, SIGNAL(timeout()),
				mapper, SLOT(map()));
		mapper->setMapping(&buffers[i].renderTimer, i);
	}
	connect(mapper, SIGNAL(mapped(int)),
			this, SLOT(continueDrawing(int)));
}

void Viewport::initBuffers()
{
	/* (re)set framebuffers */
	target->makeCurrent();

	if(!tryInitBuffers()) {
		// tryInitBuffers() fails if the buffer format is not supported by the
		// GL driver. RGBA8 is the default and should be supported by all
		// drivers.

		// TODO: write operator<<
		const char* bfs;
		switch(bufferFormat) {
		case RGBA8: bfs = "RGBA8"; break;
		case RGBA16F: bfs = "RGBA16F"; break;
		case RGBA32F: bfs = "RGBA32F"; break;
		default:
			bfs = "UNKNOWN_BUFFER_FORMAT"; break;
		}

		std::cerr << "Viewport: Warning: Failed to init framebuffer "
				  << "with format " << bfs << ". "
				  << "Falling back to RGBA8. Drawing quality reduced."
				  << std::endl;

		bufferFormat = RGBA8;
		// try again
		if(!tryInitBuffers()) {
			std::cerr << "Viewport: Error: Failed to init framebuffer "
					  << "with format RGBA8. Aborting."
					  << std::endl;
			std::abort();
		}
	}
}

bool Viewport::tryInitBuffers()
{
	for (int i = 0; i < 2; ++i) {
		// first: formats
		QGLFramebufferObjectFormat format_buf, format_blit;

		/* use floating point for better alpha accuracy in back buffer! */
		if (i == 0)
			format_blit.setInternalTextureFormat(bufferFormat);

		// strict: format_buf must be format_blit + multisampling! OGL spec!
		format_buf = format_blit;

		/* multisampling. 0 deactivates. 2 or 4 should be reasonable values.
		 * TODO: make this configurable and/or test with cheap GPUs */
		format_buf.setSamples(4);

		// second: buffer
		delete buffers[i].fbo;
		buffers[i].fbo = new QGLFramebufferObject(width, height, format_buf);
		if(!buffers[i].fbo->isValid()) {
			return false;
		}
		buffers[i].dirty = true;

		// third: blit buffer
		delete buffers[i].blit;
		buffers[i].blit = new QGLFramebufferObject(width, height, format_blit);
		if(!buffers[i].blit->isValid()) {
			return false;
		}
	}
	return true;
}

/********* S T A T E ********/

void Viewport::drawBackground(QPainter *painter, const QRectF &rect)
{
	// update geometry
	int nwidth = painter->device()->width();
	int nheight = painter->device()->height();
	if (nwidth != width || nheight != height) {
		width = nwidth;
		height = nheight;

		// update transformation (needed already for legend, axes drawing)
		updateModelview();

		// defer buffer update (do not hinder resizing with slow update)
		buffers[0].dirty = buffers[1].dirty = true;
		resizeTimer.start(150);
	}

	// draw
	drawScene(painter);
}

void Viewport::resizeScene()
{
	// initialize buffers with new size
	initBuffers();

	// update buffers
	updateBuffers();
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

void Viewport::rebuild()
{
	// guess: we want the lock to carry over both methods..
	SharedDataLock ctxlock(ctx->mutex);
	SharedDataLock setslock(sets->mutex);

	prepareLines(); // will also call reset() if indicated by ctx
	updateBuffers();
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
										 drawMeans, illuminantAppl);

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
			" report@gerbilvis.org. Thank you for your help!").arg(success));
		return;
	}
}

void Viewport::activate()
{
	if (!active) {
		active = true;
		emit activated(type);
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
	update();
}

void Viewport::insertPixelOverlay(const QPolygonF &points)
{
	overlayPoints = points;
	overlayMode = true;
	update();
}

void Viewport::changeIlluminantCurve(QVector<multi_img::Value> illum)
{
	illuminantCurve = illum;
	if (illuminant_show)
		update();
}

void Viewport::setIlluminationCurveShown(bool show)
{
	illuminant_show = show;
	update();
}

void Viewport::setAppliedIlluminant(QVector<multi_img_base::Value> illum)
{
	//bool change = (applied != illuminant_apply);
	illuminantAppl = illum.toStdVector();
/*	if (change) TODO: I assume this is already triggered by invalidated ROI
		rebuild();*/
}

void Viewport::setLimiters(int label)
{
	if (label < 1) {	// not label
		SharedDataLock ctxlock(ctx->mutex);
		limiters.assign((*ctx)->dimensionality,
						std::make_pair(0, (*ctx)->nbins-1));
		if (label == -1) {	// use hover data
			int b = selection;
			int h = hover;
			limiters[b] = std::make_pair(h, h);
		}
	} else {            // label holds data
		SharedDataLock setslock(sets->mutex);
		if ((int)(*sets)->size() > label && (**sets)[label].totalweight > 0) {
			// use range from this label
			const std::vector<std::pair<int, int> > &b =
					(**sets)[label].boundary;
			limiters.assign(b.begin(), b.end());
		} else {
			setLimiters(0);
		}
	}
}

void Viewport::highlightSingleLabel(int index)
{
	highlightLabel = index;
	updateBuffers(Viewport::RM_STEP,
				   (highlightLabel > -1 ? RM_SKIP : RM_STEP));
}

void Viewport::setAlpha(float alpha)
{
	useralpha = alpha;
	updateBuffers(Viewport::RM_STEP, Viewport::RM_SKIP);
}

void Viewport::setLimitersMode(bool enabled)
{
	limiterMode = enabled;
	updateBuffers(Viewport::RM_SKIP, Viewport::RM_STEP);
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

void Viewport::startNoHQ()
{
	drawingState = (drawingState == HIGH_QUALITY ? HIGH_QUALITY_QUICK : QUICK);
}

bool Viewport::endNoHQ(RenderMode spectrum, RenderMode highlight)
{
	/* we enter this function after the display was in temporary low quality
	 * state. it may be "dirty", i.e. need to be redrawn in high quality now
	 * that is, if the global setting is not low quality anyways
	 */
	bool dirty = drawHQ;
	if (drawingState == HIGH_QUALITY || drawingState == HIGH_QUALITY_QUICK)
		dirty = false;

	drawingState = (drawHQ ? HIGH_QUALITY : QUICK);
	if (dirty)
		updateBuffers(spectrum, highlight);
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

void Viewport::toggleLabeled(bool enabled)
{
	showLabeled = enabled;
	updateBuffers();
}

void Viewport::toggleUnlabeled(bool enabled)
{
	showUnlabeled = enabled;
	updateBuffers();
}

void Viewport::screenshot()
{
	// ensure high quality
	drawingState = HIGH_QUALITY;
	updateBuffers(RM_FULL, RM_FULL);

	// render into our buffer
	QGLFramebufferObject b(width, height);
	QPainter p(&b);
	bool success = drawScene(&p, false);

	// reset drawing state
	endNoHQ();

	if (!success)
		return;

	QImage img = b.toImage();

	// write out
	cv::Mat output = vole::QImage2Mat(img);
	IOGui io("Screenshot File", "screenshot", target);
	io.writeFile(QString(), output);
}


