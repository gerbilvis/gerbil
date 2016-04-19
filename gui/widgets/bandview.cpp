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
#include <QGraphicsPixmapItem>
#include <QKeyEvent>
#include <QSettings>
#include <QApplication>
#include <opencv2/imgproc/imgproc.hpp> // for createCursor()
#include <iostream>
#include <cmath>
#include <tbb/task.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include "qmath.h"

#include "gerbil_gui_debug.h"
#include <QDebug>

BandView::BandView()
    : // note: start with invalid curLabel to trigger proper initialization!
      cacheValid(false), cursor(-1, -1), lastcursor(-1, -1), curLabel(-1),
      overlay(0), showLabels(true), selectedLabels(0),
      ignoreUpdates(false), labelAlpha(63),
      seedColors(std::make_pair(
                     QColor(255, 0, 0, 255), QColor(255, 255, 0, 255)))
{
	// the timer automatically sends an accumulated update request
	labelTimer.setSingleShot(true);
	labelTimer.setInterval(500);

	connect(QApplication::instance(), SIGNAL(lastWindowClosed()),
	        this, SLOT(saveState()));

	initCursors();
}

void BandView::initUi()
{
	connect(&labelTimer, SIGNAL(timeout()),
	        this, SLOT(commitLabelChanges()));

	restoreState(); // note: needs setAlphaValue() signal connected
}

void BandView::toggleCursorMode()
{
	cursorMode = (cursorMode == CursorMode::Marker ? CursorMode::Rubber
												   : CursorMode::Marker);
}

void BandView::toggleOverrideMode()
{
	overrideMode = (overrideMode == OverrideMode::On ? OverrideMode::Off
													 : OverrideMode::On);
}

void BandView::updateCursorSize(CursorSize size)
{
	cursorSize = size;
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
	bool drawCursor = (inputMode == InputMode::Label ||
	                   inputMode == InputMode::Seed);
	if (!pixmap.rect().contains(cursor.x(), cursor.y()))
		drawCursor = false;
	if (curLabel < 1 || curLabel >= labelColors.count())
		drawCursor = false;
	if (drawCursor) {

		QPen pen;
		if (inputMode == InputMode::Seed) {
			pen = QPen(Qt::yellow);
		} else if (cursorMode == CursorMode::Marker) {
			pen = QPen(labelColors.at(curLabel));
		} else if (cursorMode == CursorMode::Rubber) {
			pen = QPen(QColor(Qt::white));
		}

		pen.setWidth(0);
		painter->setPen(pen);

		painter->drawConvexPolygon(getCursorHull(cursor.x(), cursor.y()));
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

	if (inputMode == InputMode::Seed) {
		pen.setColor(Qt::yellow); pen.setWidthF(1.5); pen.setStyle(Qt::DotLine);
		painter->setPen(pen);
		painter->setBrush(Qt::NoBrush);
		painter->drawRect(pixmap.rect());
	}

	painter->restore();
}

void BandView::updateCache()
{
	cachedPixmap = pixmap.copy();
	cacheValid = true;
	if (inputMode != InputMode::Seed && !showLabels) // there is no overlay, leave early
		return;

	QPainter painter(&cachedPixmap);
	//	painter.setCompositionMode(QPainter::CompositionMode_Darken);

	QImage dest(pixmap.width(), pixmap.height(), QImage::Format_ARGB32);

	tbb::parallel_for(tbb::blocked_range2d<size_t>(
	                      0, pixmap.height(), 0, pixmap.width()),
	                  [&](tbb::blocked_range2d<size_t> r) {
		for (size_t y = r.rows().begin(); y != r.rows().end(); ++y) {
			const short *lrow = labels[y], *srow = seedMap[y];
			QRgb *destrow = (QRgb*)dest.scanLine(y);
			for (size_t x = r.cols().begin(); x != r.cols().end(); ++x) {
				short lval = lrow[x], sval = srow[x];
				destrow[x] = qRgba(0, 0, 0, 0);
				if (inputMode == InputMode::Seed) {
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
	});

	painter.drawImage(0, 0, dest);
}

// helper to color single pixel with labeling
void BandView::markCachePixel(QPainter &p, int x, int y)
{
	if (inputMode != InputMode::Label)
		return;

	short l = labels(y, x);
	if (l > 0) {
		p.setPen(labelColorsA.at(l));
		p.drawPoint(x, y);
	}
}

// helper to color single pixel in seed mode
void BandView::markCachePixelS(QPainter &p, int x, int y)
{
	if (inputMode != InputMode::Seed)
		return;

	short l = seedMap(y, x);
	if (l < 64 || l > 192) {
		p.setPen(l < 64 ? seedColors.first : seedColors.second);
		p.drawPoint(x, y);
	}
}

void BandView::updateCache(int y, int x, short label)
{
	if (inputMode != InputMode::Label &&
	    inputMode != InputMode::Seed)
		return;

	if (!cacheValid) {
		updateCache();
		return;
	}

	QPixmap &p = cachedPixmap;
	QPainter painter(&p);
	// restore pixel
	painter.drawPixmap(x, y, pixmap, x, y, 1, 1);

	if (inputMode != InputMode::Seed && !showLabels) // there is no overlay, leave early
		return;

	// if needed, color pixel
	const QColor *col = 0;
	short val = (inputMode == InputMode::Seed ? seedMap(y, x) : label);
	if (inputMode == InputMode::Seed) {
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
	//Stopwatch s("Overlay drawing");
	overlay = &mask;
	update();
}

void BandView::cursorAction(QGraphicsSceneMouseEvent *ev, bool click)
{
	ScaledView::cursorAction(ev, click);

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

	// test for dimension match
	if (!pixmap.rect().contains(cursor)) {
		lastcursor = QPoint(-1,-1);
		return;
	}

	if (click) {
		// invalidate lastcursor
		lastcursor = QPoint(-1, -1);
	} else {
		// no click and no significant movement
		if (cursor == lastcursor)
			return;
	}

	// lastcursor invalid -> begin drawing at cursor
	if (QPoint(-1, -1) == lastcursor) {
		lastcursor = cursor;
	}

	/// single label case
	if (inputMode == InputMode::Pick && showLabels) {
		short cursorLabel = labels(cursor.y(), cursor.x());
		if (ev->button() & Qt::LeftButton) {
			toggleLabelHighlight(cursorLabel);
			emit labelSelected(curLabel);
		} else {
			drawOverlay(curMask);
		}

		return;
	}
	/// end of function for singleLabel case. no manipulations,
	/// destroying overlay etc.

	if (ev->buttons() != Qt::NoButton &&
	    (inputMode == InputMode::Label || inputMode == InputMode::Seed)) {
		/* alter all pixels on the line between previous and current position.
		 * the idea is that due to lag we might not get a notification about
		 * every pixel the mouse moved over. this is a good approximation. */
		QLineF line(lastcursor, cursor);
		qreal step = 1 / line.length();
		for (qreal t = 0.0; t <= 1.0; t += step) {
			QPointF point = line.pointAt(t);
			int x = point.x();
			int y = point.y();

			if (ev->buttons() & Qt::LeftButton) {
				if (inputMode == InputMode::Seed) {
					for (QPointF p : getCursor(x, y)) {
						seedMap(p.y(), p.x()) = 0;
						updateCache(p.y(), p.x());
					}
				} else {
					for (QPointF p : getCursor(x, y)) {
						updatePixel(p.x(), p.y());
					}
				}
			} else if (ev->buttons() & Qt::RightButton) {
				if (inputMode == InputMode::Seed) {
					for (QPointF p : getCursor(x, y)) {
						seedMap(p.y(), p.x()) = 255;
						updateCache(p.y(), p.x());
					}
				}
			}
		}
		if (inputMode != InputMode::Seed)
			// we must have updated something in the labeling
			labelTimer.start();
	}

	if (!grandupdate)
		updatePoint(lastcursor); // show change. Why is lastcursor enough?
	lastcursor = cursor;

	if (grandupdate) {
		overlay = NULL;
		update();
	}
}

void BandView::updatePixel(int x, int y)
{
	if (!pixmap.rect().contains(x, y))
		return;

	if (cursorMode == CursorMode::Marker) {
		if (overrideMode == OverrideMode::On
		    || (overrideMode == OverrideMode::Off && (labels(y,x) == 0))) {
			uncommitedLabels(y, x) = 1;
			labels(y, x) = curLabel;
			updateCache(y, x, curLabel);
		}
	} else if (cursorMode == CursorMode::Rubber) {	
		if (overrideMode == OverrideMode::On
		    || (overrideMode == OverrideMode::Off && (labels(y,x) == curLabel))) {
			uncommitedLabels(y, x) = 1;
			labels(y, x) = 0;
			updateCache(y, x, 0);
		}
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

	ScaledView::leaveEvent();
}

void BandView::keyPressEvent(QKeyEvent *event)
{
	switch (event->key()) {
	case Qt::Key_C:
		if (inputMode == InputMode::Seed)
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
	if (enabled) {
		lastMode = inputMode;
		inputMode = InputMode::Seed;
	} else {
		inputMode = lastMode;
	}
	emit inputModeChanged(inputMode);
	refresh();
}

void BandView::toggleShowLabels(bool disabled)
{
	if (showLabels == disabled) {	// i.e. not the state we want
		showLabels = !showLabels;
		refresh();
	}
}

void BandView::toggleLabelHighlight(short label)
{
	bool toDelete = false;
	if (selectedLabels.contains(label)) {
		toDelete = true;
		//qDebug() << "REMOVED LABEL";
	} else {
		selectedLabels.push_back(label);
		toDelete = false;
	}
	curLabel = label;

	if (selectedLabels.size() == 1) {
		curMask = cv::Mat1b(labels.rows, labels.cols, (uchar)0);
	} else {
		curMask = curMask | cv::Mat1b(labels.rows, labels.cols, (uchar)0);
	}

	if (toDelete) {
		curMask.setTo(0, (labels == label));
		int pos = selectedLabels.indexOf(label);
		selectedLabels.remove(pos, 1);
	} else {
		curMask.setTo(1, (labels == label));
	}

	drawOverlay(curMask);

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

std::pair<QPolygonF, QPolygonF>
BandView::createCursor(const cv::Mat1b &mask, const QPoint &center)
{
	QPolygonF cursor, hull;
	std::vector<cv::Point> points;
	for (int y = 0; y < mask.rows; y++) {
		for (int x = 0; x < mask.cols; x++) {
			if (mask(y, x) == 1) {
				cursor.push_back(QPointF(x - center.x(), y - center.y()));
				points.push_back(cv::Point(x - center.x(), y - center.y()));
			}
		}
	}

	std::vector<int> indices;
	cv::convexHull(points, indices);
	for (auto i : indices) {
		auto p = points[i];
		hull.push_back(QPointF(p.x, p.y));
	}

	return std::make_pair(cursor, hull);
}

void BandView::initCursors()
{
	cv::Mat1b mask = (cv::Mat1b(1,1) << 1);
	QPoint center(0,0);
	cursors[CursorSize::Small] = createCursor(mask, center);

	mask = (cv::Mat1b(4,4) <<
	        0,1,1,0,
	        1,1,1,1,
	        1,1,1,1,
	        0,1,1,0);
	center = QPoint(1,1);
	cursors[CursorSize::Medium] = createCursor(mask, center);

	mask = (cv::Mat1b(7,7) <<
	        0,0,1,1,1,0,0,
	        0,1,1,1,1,1,0,
	        1,1,1,1,1,1,1,
	        1,1,1,1,1,1,1,
	        1,1,1,1,1,1,1,
	        0,1,1,1,1,1,0,
	        0,0,1,1,1,0,0);
	center = QPoint(3,3);
	cursors[CursorSize::Big] = createCursor(mask, center);

	mask = (cv::Mat1b(15,15) <<
	        0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,
	        0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,
	        0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,
	        0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
	        0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
	        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	        0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
	        0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
	        0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,
	        0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,
	        0,0,0,0,0,1,1,1,1,1,0,0,0,0,0);
	center = QPoint(7,7);
	cursors[CursorSize::Huge] = createCursor(mask, center);
}

QPolygonF BandView::getCursor(int xpos, int ypos)
{
	CursorSize size;
	if (inputMode == InputMode::Seed)
		size = CursorSize::Medium;
	else
		size = cursorSize;
	return cursors[size].first.translated(xpos, ypos);
}

QPolygonF BandView::getCursorHull(int xpos, int ypos)
{
	CursorSize size;
	if (inputMode == InputMode::Seed)
		size = CursorSize::Medium;
	else
		size = cursorSize;
	return cursors[size].second.translated(xpos + 0.5f, ypos + 0.5f);
}

QMenu *BandView::createContextMenu()
{
	QMenu* contextMenu = ScaledView::createContextMenu();

	contextMenu->addSeparator();
	contextMenu->addAction(zoomAction);
	contextMenu->addAction(labelAction);
	contextMenu->addAction(pickAction);

	return contextMenu;
}

void BandView::saveState()
{
	QSettings settings;
	settings.setValue("BandView/alphaValue", labelAlpha);
}

void BandView::restoreState()
{
	QSettings settings;
	auto alphaValue = settings.value("BandView/alphaValue", 63);
	emit setAlphaValue(alphaValue.toInt());
}
