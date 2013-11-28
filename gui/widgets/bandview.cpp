/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "widgets/bandview.h"

#include <stopwatch.h>
#include <QPainter>
#include <QGraphicsSceneEvent>
#include <QKeyEvent>
#include <iostream>
#include <cmath>
#include <tbb/task.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include "gerbil_gui_debug.h"

// for debugging
std::ostream& operator<<(std::ostream& stream, const QPointF &p) {
	stream << "(" << p.x() << "," << p.y() << ")";
	return stream;
}

BandView::BandView()
	: // note: start with invalid curLabel to trigger proper initialization!
	  cacheValid(false), cursor(-1, -1), lastcursor(-1, -1), curLabel(-1),
	  overlay(0), showLabels(true), singleLabel(false), holdLabel(false),
	  ignoreUpdates(false),
	  seedMode(false), labelAlpha(63),
	  seedColors(std::make_pair(
			QColor(255, 0, 0, 255), QColor(255, 255, 0, 255)))
{
	// the timer automatically sends an accumulated update request
	labelTimer.setSingleShot(true);
	labelTimer.setInterval(500);
}

void BandView::initUi()
{
	connect(&labelTimer, SIGNAL(timeout()),
			this, SLOT(commitLabelChanges()));
}

void BandView::setPixmap(QPixmap p)
{
	ScaledView::setPixmap(p);

	// adjust seed map if necessary
	if (seedMap.empty()
		|| seedMap.rows != pixmap.height() || seedMap.cols != pixmap.width())
	{
		seedMap = cv::Mat1s(pixmap.height(), pixmap.width(), (short)127);
		// TODO: send signal to model, maybe move whole seed map to model
	}

	refresh();
}

void BandView::refresh()
{
	cacheValid = false;
	update();
}

void BandView::updateLabeling(const cv::Mat1s &newLabels,
							  const QVector<QColor> &colors,
							  bool colorsChanged)
{
	if (ignoreUpdates)
		return;

	if (!colors.empty()) {
		// store label colors
		labelColors = colors;

		// create alpha-modified label colors
		labelColorsA.resize(colors.size());
		labelColorsA[0] = QColor(0, 0, 0, 0);
		for (int i = 1; i < labelColors.size(); ++i) // 0 is index for unlabeled
		{
			QColor col = labelColors[i];
			col.setAlpha(labelAlpha);
			labelColorsA[i] = col;
		}
	}

	if (!newLabels.empty()) {
		// local labeling is a copy
		labels = newLabels.clone();
		// initialize mask for uncommited labels with right dimensions
		uncommitedLabels = cv::Mat1b(labels.rows, labels.cols, (uchar)0);
	}

	if (!newLabels.empty() || colorsChanged)
		refresh();
}

void BandView::updateLabeling(const cv::Mat1s &newLabels, const cv::Mat1b &mask)
{
	if (ignoreUpdates)
		return;

	// only incorporate pixels in mask, as we may have local updates as well
	newLabels.copyTo(labels, mask);

	refresh();
}

void BandView::paintEvent(QPainter *painter, const QRectF &rect)
{
	/* deal with no pixmap set (width==0), or labeling does not fit the pixmap
	 * (as updates to pixmap and labeling are not synchronised) */
	bool consistent = ((pixmap.width() == labels.cols) &&
					   (pixmap.height() == labels.rows));

	if (!consistent) {
		painter->fillRect(rect, QBrush(Qt::gray, Qt::BDiagPattern));
		drawWaitMessage(painter);
		return;
	}

	if (!cacheValid) {
		updateCache();
	}

	fillBackground(painter, rect);

	painter->save();

	//painter.setRenderHint(QPainter::Antialiasing); too slow!
	painter->setWorldTransform(scaler);
	QRectF damaged = scalerI.mapRect(rect);
	painter->drawPixmap(damaged, cachedPixmap, damaged);

	/* draw current cursor */
	bool drawCursor = (seedMode || !singleLabel);
	if (!pixmap.rect().contains(cursor.x(), cursor.y()))
		drawCursor = false;
	if (curLabel < 1 || curLabel >= labelColors.count())
		drawCursor = false;

	if (drawCursor) {
//		GGDBGM(boost::format("count=%1% curLabel=%2%")
//			  %labelColors.count()% curLabel << endl);
		QPen pen(seedMode ? Qt::yellow : labelColors.at(curLabel));
		pen.setWidth(0);
		painter->setPen(pen);
		painter->drawRect(QRectF(cursor, QSizeF(1, 1)));
	}

	/* draw overlay (a quasi one-timer) */
	QPen pen;
	if (overlay) {
		QImage dest(overlay->cols, overlay->rows, QImage::Format_ARGB32);
		dest.fill(qRgba(0, 0, 0, 0));
		for (int y = 0; y < overlay->rows; ++y) {
			const unsigned char *srcrow = (*overlay)[y];
			QRgb *destrow = (QRgb*)dest.scanLine(y);
			for (int x = 0; x < overlay->cols; ++x) {
				if (srcrow[x])
					destrow[x] = qRgba(255, 255, 0, 255);
			}
		}
		painter->drawImage(0, 0, dest);
	}

	if (seedMode) {
		/* TODO: most times barely visible! */
		pen.setColor(Qt::yellow); pen.setWidthF(0.5); pen.setStyle(Qt::DotLine);
		painter->setPen(pen);
		painter->setBrush(Qt::NoBrush);
		painter->drawRect(0, 0, pixmap.width(), pixmap.height());
	}

	painter->restore();
}

struct updateCacheBody {
	QImage &dest;
	bool seedMode;
	const cv::Mat1s &labels;
	const cv::Mat1s &seedMap;
	const QVector<QColor> &labelColorsA;
	const std::pair<QColor, QColor> &seedColors;

	updateCacheBody(QImage &dest, bool seedMode, const cv::Mat1s &labels, const cv::Mat1s &seedMap,
		const QVector<QColor> &labelColorsA, const std::pair<QColor, QColor> &seedColors)
		: dest(dest), seedMode(seedMode), labels(labels), seedMap(seedMap),
		labelColorsA(labelColorsA), seedColors(seedColors) {}

	void operator()(const tbb::blocked_range2d<size_t> &r) const {
		for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
			const short *lrow = labels[y], *srow = seedMap[y];
			QRgb *destrow = (QRgb*)dest.scanLine(y);
			for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
				short lval = lrow[x], sval = srow[x];
				destrow[x] = qRgba(0, 0, 0, 0);
				if (seedMode) {
					if (sval == 255)
						destrow[x] = seedColors.first.rgba();
					else if (sval == 0)
						destrow[x] = seedColors.second.rgba();
					else if (lval > 0)
						destrow[x] = labelColorsA[lval].rgba();
				} else if (lval > 0) {
					destrow[x] = labelColorsA[lval].rgba();
				}
			}
		}
	}
};

void BandView::updateCache()
{
	cachedPixmap = pixmap.copy();
	cacheValid = true;
	if (!seedMode && !showLabels) // there is no overlay, leave early
		return;

	QPainter painter(&cachedPixmap);
//	painter.setCompositionMode(QPainter::CompositionMode_Darken);

	QImage dest(pixmap.width(), pixmap.height(), QImage::Format_ARGB32);
	updateCacheBody body(dest, seedMode, labels, seedMap, labelColorsA, seedColors);
	tbb::parallel_for(tbb::blocked_range2d<size_t>(
		0, pixmap.height(), 0, pixmap.width()), body);

	painter.drawImage(0, 0, dest);
}

// helper to color single pixel with labeling
void BandView::markCachePixel(QPainter &p, int x, int y)
{
	short l = labels(y, x);
	if (l > 0) {
		p.setPen(labelColorsA.at(l));
		p.drawPoint(x, y);
	}
}

// helper to color single pixel in seed mode
void BandView::markCachePixelS(QPainter &p, int x, int y)
{
	short l = seedMap(y, x);
	if (l < 64 || l > 192) {
		p.setPen(l < 64 ? seedColors.first : seedColors.second);
		p.drawPoint(x, y);
	}
}

void BandView::updateCache(int y, int x, short label)
{
	if (!cacheValid) {
		updateCache();
		return;
	}

	QPixmap &p = cachedPixmap;
	QPainter painter(&p);
	// restore pixel
	painter.drawPixmap(x, y, pixmap, x, y, 1, 1);

	if (!seedMode && !showLabels) // there is no overlay, leave early
		return;
	
	// if needed, color pixel
	const QColor *col = 0;
	short val = (seedMode ? seedMap(y, x) : label);
	if (seedMode) {
		if (val == 255)
			col = &seedColors.first;
		else if (val == 0)
			col = &seedColors.second;
	} else if (val > 0) {
		col = &labelColorsA.at(val);
	}

	if (col) {
		painter.setPen(*col);
		painter.drawPoint(x, y);
	}
}

void BandView::drawOverlay(const cv::Mat1b &mask)
{
	//vole::Stopwatch s("Overlay drawing");
	overlay = &mask;
	update();
}

void BandView::cursorAction(QGraphicsSceneMouseEvent *ev, bool click)
{
	bool consistent = ((pixmap.width() == labels.cols) &&
						   (pixmap.height() == labels.rows));
	if (!consistent) // not properly initialized
		return;

	// kill overlay to free the view
	bool grandupdate = (overlay != NULL);

	// do the mapping in floating point for accuracy
	QPointF cursorF = scalerI.map(QPointF(ev->scenePos()));
	cursor.setX(std::floor(cursorF.x() - 0.25));
	cursor.setY(std::floor(cursorF.y() - 0.25));

	//GGDBGM(boost::format("lastcursor %1%, cursor %2%, clicked %3%")%lastcursor%cursor%click << endl);

	// invalidate lastcursor
	if (click) {
		lastcursor = QPoint(-1, -1);
	}

	// nothing new after all..
	if (cursor == lastcursor)
		return;

	// lastcursor invalid -> begin drawing at cursor
	if (QPoint(-1, -1) == lastcursor) {
		lastcursor = cursor;
	}

	// test for dimension match
	if (!pixmap.rect().contains(cursor)) {
		lastcursor = QPoint(-1,-1);
		return;
	}

	int x = cursor.x(), y = cursor.y();
	if (singleLabel && showLabels) {
		if (ev->buttons() & Qt::LeftButton) {
			holdLabel = !holdLabel;
		}
		if (!holdLabel && (labels(y, x) != curLabel)) {
			curLabel = labels(y, x);
			curMask = cv::Mat1b(labels.rows, labels.cols, (uchar)0);
			curMask.setTo(1, (labels == curLabel));
			drawOverlay(curMask);
			emit singleLabelSelected(curLabel); // dist view redraw
		} else {
			if (overlay != &curMask)
				drawOverlay(curMask);
		}
		emit pixelOverlay(y, x);
		return;
	}

	/// end of function for singleLabel case. no manipulations,
	/// destroying overlay etc.

	// overlay in spectral views but not during pixel labeling (reduce lag)
	if (ev->buttons() == Qt::NoButton)
		emit pixelOverlay(y, x);

	if (ev->buttons() != Qt::NoButton) {
		/* alter all pixels on the line between previous and current position.
		 * the idea is that due to lag we might not get a notification about
		 * every pixel the mouse moved over. this is a good approximation. */
		QLineF line(lastcursor, cursor);
		qreal step = 1 / line.length();
		for (qreal t = 0.0; t < 1.0; t += step) {
			QPointF point = line.pointAt(t);
			int x = point.x();
			int y = point.y();

			if (!pixmap.rect().contains(x, y))
				break;

			if (ev->buttons() & Qt::LeftButton) {
				if (!seedMode) {
					uncommitedLabels(y, x) = 1;
					labels(y, x) = curLabel;
					updateCache(y, x, curLabel);
					labelTimer.start();
				} else {
					seedMap(y, x) = 0;
					updateCache(y, x);
				}
			// erase
			} else if (ev->buttons() & Qt::RightButton) {
				if (!seedMode) {
					if (labels(y, x) == curLabel) {
						uncommitedLabels(y, x) = 1;
						labels(y, x) = 0;
						updateCache(y, x, 0);
						labelTimer.start();
						if (!grandupdate)
							updatePoint(cursor);
					}
				} else {
					seedMap(y, x) = 255;
					updateCache(y, x);
				}
			}
		}
	}

	if (!grandupdate)
		updatePoint(lastcursor);
	lastcursor = cursor;

	if (grandupdate) {
		overlay = NULL;
		update();
	}
}

void BandView::commitLabelChanges()
{
	if (cv::sum(uncommitedLabels)[0] == 0)
		return; // label mask empty

	ignoreUpdates = true;
	emit alteredLabels(labels, uncommitedLabels);
	ignoreUpdates = false;

	/* it is important to create new matrix. Otherwise changes would overwrite
	 * the matrix we just sent out in a signal. OpenCV… */
	uncommitedLabels = cv::Mat1b(labels.rows, labels.cols, (uchar)0);
	labelTimer.stop();
}

void BandView::commitLabels()
{
	ignoreUpdates = true;
	emit newLabeling(labels);
	ignoreUpdates = false;
}

void BandView::updatePoint(const QPoint &p)
{
	QPointF damagetl = scaler.map(QPointF(p.x() - 2, p.y() - 2));
	QPointF damagebr = scaler.map(QPointF(p.x() + 2, p.y() + 2));
	// force redraw
	invalidate(QRectF(damagetl, damagebr), BackgroundLayer);
}

void BandView::clearSeeds()
{
	seedMap.setTo(127);
	refresh();
}

void BandView::enterEvent()
{
	/* as we have some action in the window that is only triggered by hovering,
	 * we steal keyboard focus here. Otherwise the user would have to click
	 * first just to get keyboard focus (unintuitive)
	 */
	setFocus(Qt::MouseFocusReason);
}

void BandView::leaveEvent()
{
//	GGDBGM("leaveEvent" << endl);
	// invalidate cursor
	cursor = lastcursor = QPoint(-1, -1);

	// invalidate previous overlay
	emit pixelOverlay(-1, -1);
	update();
}

void BandView::keyPressEvent(QKeyEvent *event)
{
	switch (event->key()) {
	case Qt::Key_C:
		if (seedMode)
			clearSeeds();
		else
			emit clearRequested();
		break;
	default:
		ScaledView::keyPressEvent(event);
	}
}

void BandView::setCurrentLabel(int label)
{
	curLabel = label;
}

void BandView::toggleSeedMode(bool enabled)
{
	seedMode = enabled;
	refresh();
}

void BandView::toggleShowLabels(bool disabled)
{
	if (showLabels == disabled) {	// i.e. not the state we want
		showLabels = !showLabels;
		refresh();
	}
}

void BandView::toggleSingleLabel(bool enabled)
{
	if (singleLabel != enabled) {	// i.e. not the state we want
		singleLabel = !singleLabel;
		refresh();
	}
}

void BandView::highlightSingleLabel(short label, bool highlight)
{
	if (highlight) {
		curMask = cv::Mat1b(labels.rows, labels.cols, (uchar)0);
		curMask.setTo(1, (labels == label));
		drawOverlay(curMask);
	} else {
		overlay = NULL;
		update();
	}
}

void BandView::applyLabelAlpha(int alpha)
{
	if (labelAlpha == alpha)
		return;

	labelAlpha = alpha;
	for (int i = 1; i < labelColorsA.size(); ++i)
		labelColorsA[i].setAlpha(labelAlpha);

	refresh();
}


void BandView::setSeedMap(cv::Mat1s seeding)
{
	seedMap = seeding;
	refresh();
}
