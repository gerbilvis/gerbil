#include "viewport.h"

#include <app/gerbilapplication.h>

#include <QGraphicsProxyWidget>
#include <QMessageBox>
#include <QDebug>
#include <QAction>
#include <limits>
#include <algorithm>

#include <gerbil_gui_debug.h>

bool Viewport::drawScene(QPainter *painter, bool withDynamics)
{
	bool disabled = false;
	{
		/* TODO: disabled member state instead? */
		SharedDataLock ctxlock(ctx->mutex);
		SharedDataLock setslock(sets->mutex);
		if ((*sets)->empty() || (*ctx)->wait)
			disabled = true;
	}

	target->makeCurrent();

	/* draw background */
	QRect rect(0, 0, width, height);
	/* Hack: without dynamics, we typically also want a boring background */
	if (withDynamics)
		painter->fillRect(rect, QColor(15, 7, 15));
	else
		painter->fillRect(rect, Qt::black);
	painter->setRenderHint(QPainter::Antialiasing);

	if (disabled) {
		drawWaitMessage(painter);
		return false;
	}

	painter->save();
	painter->setWorldTransform(modelview);
	drawAxesBg(painter);
	painter->restore();

	/* determine if we draw the highlight part */
	// only draw when active and dynamic content is desired
	bool drawHighlight = active && withDynamics;
	// only draw if not implicitely empty
	drawHighlight = drawHighlight && (hover > -1 || limiterMode);
	// do not draw when in single pixel overlay mode
	drawHighlight = drawHighlight && (!overlayMode);
	// do not draw when in single label mode
	drawHighlight = drawHighlight && (highlightLabels.empty());

	for (int i = 0; i < (drawHighlight ? 2 : 1); ++i) {

		renderbuffer &b = buffers[i];
		if (b.dirty) {
			drawWaitMessage(painter);
			// nothing to draw yet, don't even bother with other buffer,
			disabled = true;
			break;
		}

		// blit first to get from multisample to regular buffer. then draw that
		QGLFramebufferObject::blitFramebuffer(b.blit, rect, b.fbo, rect);
		target->drawTexture(rect, b.blit->texture());
	}

	// only provide selection for view with dynamic components (selected band)
	if (withDynamics)
		drawLegend(painter, selection);
	else
		drawLegend(painter);

	// foreground axes are a dynamic part
	if (withDynamics) {
		painter->save();
		painter->setWorldTransform(modelview);
		drawAxesFg(painter);
		painter->restore();
	}

	if (overlayMode && withDynamics)
		drawOverlay(painter);

	// return success if nothing was omited
	return !disabled;
}

void Viewport::updateBuffers(RenderMode spectrum, RenderMode highlight)
{
	if (!buffers[0].fbo || !buffers[1].fbo)
		return;

	{
		SharedDataLock ctxlock(ctx->mutex);
		SharedDataLock setslock(sets->mutex);
		if ((*sets)->empty() || (*ctx)->wait)
			return;
	}

	// even if we had HQ last time, this time it will be dirty!
	if (drawingState == HIGH_QUALITY_QUICK)
		drawingState = QUICK;

	// array for convenience
	RenderMode mode[2] = { spectrum, highlight };
	QRect rect(0, 0, width, height);

	for (int i = 0; i < 2; ++i) {
		renderbuffer &b = buffers[i];

		if (mode[i] == RM_SKIP)
			continue;

		b.renderTimer.stop();
		b.renderedLines = 0;

		if (!(b.fbo->isValid() && b.blit->isValid())) {
			GerbilApplication::internalError(
			            "Framebuffer not valid in viewport updateBuffers().",
			            false);
			return;
		}

		// does not make much sense here, but seems to help with no/partial update problems
		target->makeCurrent();
		QPainter painter(b.fbo);

		painter.setCompositionMode(QPainter::CompositionMode_Source);
		painter.fillRect(rect, Qt::transparent);
		painter.setCompositionMode(QPainter::CompositionMode_SourceOver);

		if (drawingState == HIGH_QUALITY)
			painter.setRenderHint(QPainter::Antialiasing);

		painter.save();
		painter.setWorldTransform(modelview);
		drawBins(painter, b.renderTimer, b.renderedLines,
		         (mode[i] == RM_FULL) ? std::numeric_limits<int>::max()
		                              : b.renderStep, (i == 1));
		painter.restore();
		b.dirty = false;
	}

	update();
}

void Viewport::updateYAxis(bool yAxisChanged)
{
	// steps on the y-axis (number of labeled values)
	const int amount = 10;

	/* calculate raw numbers for y-axis */
	std::vector<float> ycoord(amount);
	float maximum = 0.f;

	SharedDataLock ctxlock(ctx->mutex);
	float plotmaxval = (*ctx)->maxval;
	float plotminval = (*ctx)->minval;
	float binscount = (qreal)((*ctx)->nbins);
	ctxlock.unlock();

	float maxvalue;
	float range = 1/zoom;
	if (yAxisChanged) {
		QPointF bottom(0.f, boundaries.vp);
		bottom = modelviewI.map(bottom);

		qreal ratio = bottom.y()/binscount;

		maxvalue = plotmaxval - (1.f-ratio) * (plotmaxval - plotminval);
		if (maxvalue > plotmaxval)
			maxvalue = plotmaxval;
	} else {
		maxvalue = plotmaxval;
	}

	for (int i = 0; i < amount; ++i) {
		ycoord[i] = maxvalue - i*(1.f/(amount-1))*range*(plotmaxval - plotminval);
		ycoord[i] = std::max(ycoord[i], plotminval);

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

void Viewport::updateModelview(bool newBinning)
{
	SharedDataLock ctxlock(ctx->mutex);

	QPointF zero;
	if (newBinning) {
		zero = modelview.map(QPointF(0.f, 0.f));
	}

	boundaries.htp = yaxisWidth + 14;
	displayHeight = height - 2*boundaries.vp - boundaries.vtp;

	// if gradient, we discard one unit space intentionally for centering
	size_t d = (*ctx)->dimensionality
	        - ((*ctx)->type == representation::GRAD ? 0 : 1);
	qreal w = (width  - 2*boundaries.hp - boundaries.htp)/(qreal)(d); // width of one unit
	qreal h = displayHeight/(qreal)((*ctx)->nbins); // height of one unit
	int t = ((*ctx)->type == representation::GRAD ? w/2 : 0); // moving half a unit for centering

	modelview.reset();
	modelview.translate(boundaries.hp + boundaries.htp + t,
	                    boundaries.vp);
	modelview.scale(w, -1*h); // -1 low values at bottom
	modelview.translate(0, -((*ctx)->nbins)); // shift for low values at bottom

	// set inverse
	modelviewI = modelview.inverted();

	// restore previous position in plot (zoom>1 avoids initial bork transform)
	if (newBinning && zoom > 1) {
		QPointF zerolocal = modelviewI.map(zero);
		modelview.translate(zerolocal.x(),zerolocal.y());
		modelview.scale(zoom, zoom);

		// reset inverse
		modelviewI = modelview.inverted();

		// reset y-axis
		updateYAxis(true);
	}

}

void Viewport::drawBins(QPainter &painter, QTimer &renderTimer,
                        unsigned int &renderedLines, unsigned int renderStep,
                        bool highlight)
{
	SharedDataLock ctxlock(ctx->mutex);
	// TODO: this also locks shuffleIdx implicitely, better do it explicitely?
	SharedDataLock setslock(sets->mutex);

	// Stopwatch watch("drawBins");

	/* initialize painting in GL, vertex buffer */
	target->makeCurrent();
	painter.beginNativePainting();
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	bool success = vb.bind();
	if (!success) {
		GerbilApplication::internalError(
		            "Vertex buffer could not be bound in viewport drawBins().",
		            false);
		painter.endNativePainting();
		return;
	}
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(2, GL_FLOAT, 0, 0);
	size_t iD = renderedLines * (*ctx)->dimensionality;

	/* determine drawing range. could be expanded to only draw spec. labels */
	// make sure that viewport draws "unlabeled" data in ignore-label case
	int start = ((showUnlabeled || (*ctx)->ignoreLabels == 1) ? 0 : 1);
	int end = (showLabeled ? (int)(*sets)->size() : 1);

	size_t total = shuffleIdx.size();
	size_t first = renderedLines;
	size_t last = std::min((size_t)(renderedLines + renderStep), total);

	// loop over all elements in vertex index, update element and vector indices
	for (size_t i = first; i < last;
	     ++i, iD += (*ctx)->dimensionality) {
		std::pair<int, BinSet::HashKey> &idx = shuffleIdx[i];

		// filter out according to label
		bool filter = ((idx.first < start || idx.first >= end));
		// do not filter out highlighted label(s)
		if (!(*ctx)->ignoreLabels) {
			filter = filter && !highlightLabels.contains(idx.first);
		}
		if (filter) {
			// increase loop count to achieve renderStep
			if (last < total)
				++last;
			continue;
		}

		BinSet::HashKey &K = idx.second;

		// highlight mode (foreground buffer)
		if (highlight) {

			//test if we are part of the highlight
			bool highlighted = false;
			if (limiterMode) {
				highlighted = true;
				for (size_t i = 0; i < (*ctx)->dimensionality; ++i) {
					unsigned char k = K[i];
					if (k < limiters[i].first || k > limiters[i].second)
						highlighted = false;
				}
			} else if ((unsigned char)K[selection] == hover) {
				highlighted = true;
			}

			// filter out
			if (!highlighted) {
				// increase loop count to achieve renderStep
				if (last < total)
					++last;
				continue;
			}
		}

		// grab binset and bin according to key
		BinSet &s = (**sets)[idx.first];
		std::pair<BinSet::HashMap::const_iterator, BinSet::HashMap::const_iterator> binitp =
		        s.bins.equal_range(K);
		if (s.bins.end() == binitp.first) {
			// FIXME this is an error and should be treated accordingly
			GGDBGM("no bin"<< endl);
			return;
		}
		Bin const &b = s.bins.equal_range(K).first->second;

		// set color
		QColor color = determineColor((drawRGB->isChecked() ? b.rgb : s.label),
		                              b.weight, s.totalweight,
		                              highlight,
		                              highlightLabels.contains(idx.first));
		target->qglColor(color);

		// draw polyline
		glDrawArrays(GL_LINE_STRIP, (GLsizei)iD, (GLint)(*ctx)->dimensionality);
	}
	vb.release();
	painter.endNativePainting();

	// setup succeeding incremental drawing
	renderedLines += (unsigned int)(last - first);
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
	alpha = useralpha;
	if (drawLog->isChecked())
		alpha *= (0.01 + 0.99*(std::log(weight+1) / std::log(totalweight)));
	else
		alpha *= (0.01 + 0.99*(weight / totalweight));
	color.setAlphaF(std::min(alpha, 1.)); // cap at 1

	if (highlighted) {
		if (basecolor == Qt::white) {
			color = Qt::yellow;
		} else {
			color.setGreen(std::min(color.green() + 195, 255));
			color.setRed(std::min(color.red() + 195, 255));
			color.setBlue(color.blue()/2);
		}
		color.setAlphaF(1.);
	}

	// recolor singleLabel (and make 100% opaque)
	if (single) {
		color.setRgbF(1., 1., 0., 1.);
	}
	return color;
}

void Viewport::continueDrawing(int buffer)
{
	renderbuffer &b = buffers[buffer];
	if (b.dirty)
		return;

	SharedDataLock ctxlock(ctx->mutex);
	SharedDataLock setslock(sets->mutex);

	if ((*sets)->empty() || (*ctx)->wait)
		return;

	setslock.unlock();
	ctxlock.unlock();

	QPainter painter(b.fbo);

	if (drawingState == HIGH_QUALITY)
		painter.setRenderHint(QPainter::Antialiasing);

	painter.save();
	painter.setWorldTransform(modelview);
	drawBins(painter, b.renderTimer, b.renderedLines, b.renderStep,
	         (buffer == 1));
	painter.restore();

	update();
}

void Viewport::drawAxesFg(QPainter *painter)
{

	SharedDataLock ctxlock(ctx->mutex);

	if (selection < 0 || selection >= (int)(*ctx)->dimensionality)
		return;

	QPen pen;
	pen.setWidth(0); // hairline width, needed because of our projection
	pen.setColor(active ? Qt::red : Qt::gray); // draw selection in foreground
	painter->setPen(pen);

	qreal top = ((*ctx)->nbins);
	if (illuminant_show && !illuminantCurve.empty())
		top *= illuminantCurve.at(selection);
	painter->drawLine(QPointF(selection, 0.), QPointF(selection, top));

	// draw limiters
	if (limiterMode) {
		pen.setColor(Qt::red);
		painter->setPen(pen);
		for (size_t i = 0; i < (*ctx)->dimensionality; ++i) {
			qreal y1 = limiters[i].first, y2 = limiters[i].second + 1;
			if (!illuminantAppl.empty()) {
				y1 *= illuminantAppl.at(i);
				y2 *= illuminantAppl.at(i);
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
	QPen pen(QColor(64, 64, 64));
	pen.setWidth(0);
	painter->setPen(pen);

	/* without illuminant */
	if (!illuminant_show || illuminantCurve.empty()) {
		for (size_t i = 0; i < (*ctx)->dimensionality; ++i)
			painter->drawLine(i, 0, i, (*ctx)->nbins);
		return;
	}

	/* now instead with illuminant */

	// polygon describing illuminant
	QPolygonF poly;
	for (size_t i = 0; i < (*ctx)->dimensionality; ++i) {
		qreal top = ((*ctx)->nbins-1) * illuminantCurve.at(i);
		painter->drawLine(QPointF(i, 0.), QPointF(i, top));
		poly << QPointF(i, top);
	}
	poly << QPointF((*ctx)->dimensionality-1, (*ctx)->nbins-1);
	poly << QPointF(0, (*ctx)->nbins-1);

	// visualize illuminant
	QPolygonF poly2 = modelview.map(poly);
	poly2.translate(0., -5.);
	painter->restore();
	QBrush brush(QColor(32, 32, 32), Qt::Dense3Pattern);
	painter->setBrush(brush);
	painter->setPen(Qt::NoPen);
	painter->drawPolygon(poly2);
	painter->setPen(Qt::white);
	poly2.remove((int)(*ctx)->dimensionality, 2);
	painter->drawPolyline(poly2);
	painter->save();
	painter->setWorldTransform(modelview);
}

void Viewport::drawLegend(QPainter *painter, int sel)
{
	SharedDataLock ctxlock(ctx->mutex);

	assert((*ctx)->xlabels.size() == (unsigned int)(*ctx)->dimensionality);

	painter->setPen(Qt::white);

	//drawing background for x-axis
	QRectF xbgrect(0, height-35, width, 35);;

	painter->save();
	painter->setBrush(QColor(0,0,0, 128));
	painter->fillRect(xbgrect, QBrush(QColor(0,0,0,128)));
	painter->restore();

	// x-axis
	for (size_t i = 0; i < (*ctx)->dimensionality; ++i) {
		//		GGDBGM((format("label %1%: '%2%'")
		//		 %i % ((*ctx)->labels[i].toStdString()))  << endl);
		QPointF l = modelview.map(QPointF(i - 1.f, 0.f));
		l.setY(height - 30);

		QPointF r = modelview.map(QPointF(i + 1.f, 0.f));
		r.setY(height);

		QRectF rect(l, r);

		// only draw every xth label if we run out of space
		int stepping = std::max<int>(1, 150 / rect.width());

		// only draw regular steppings and selected band
		if (i % stepping && (int)i != sel)
			continue;

		// also do not draw near selected band
		if ((int)i != sel && sel > -1 && std::abs((int)i - sel) < stepping)
			continue;

		rect.adjust(-50.f, 0.f, 50.f, 0.f);

		bool highlight = ((int)i == sel);
		if (highlight)
			painter->setPen(Qt::red);
		painter->drawText(rect, Qt::AlignCenter, (*ctx)->xlabels[i]);
		if (highlight)	// revert back color
			painter->setPen(Qt::white);
	}

	//drawing background for y-axis
	QRectF ybgrect(0, 0, yaxisWidth+25, height-35);

	painter->save();
	painter->setBrush(QColor(0,0,0, 128));
	painter->fillRect(ybgrect, QBrush(QColor(0,0,0,128)));
	painter->restore();

	/// y-axis
	for (size_t i = 0; i < yaxis.size(); ++i) {
		QPointF b(0.f, (displayHeight)/(yaxis.size()-1) * i + boundaries.vp);

		QPointF t = b;
		t += QPointF(0.f, 10.f);
		t.setX(0);
		b.setX(yaxisWidth+15);
		QRectF rect(t, b);

		painter->drawText(rect, Qt::AlignVCenter | Qt::AlignRight, yaxis[i]);
	}
}

void Viewport::drawOverlay(QPainter *painter)
{
	painter->save();
	QPolygonF poly = modelview.map(overlayPoints);
	QPen pen(QColor(0, 0, 0, 127));
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

