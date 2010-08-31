#include "bandview.h"

#include <stopwatch.h>
#include <QPainter>
#include <QPaintEvent>
#include <iostream>

BandView::BandView(QWidget *parent)
	: ScaledView(parent),
	  cacheValid(false), cursor(-1, -1), lastcursor(-1, -1), curLabel(1),
	  overlay(0), showLabels(true), seedMode(false)
{}

void BandView::setPixmap(const QPixmap &p)
{
	cacheValid = false;
	pixmap = &p;
}

void BandView::paintEvent(QPaintEvent *ev)
{
	if (!pixmap)
		return;
	if (!cacheValid)
		updateCache();

	QPainter painter(this);
	//painter.setRenderHint(QPainter::Antialiasing); too slow!

	// draw band (slow!)
//	painter.drawPixmap(ev->rect(), cachedPixmap.transformed(scaler), ev->rect());

	painter.setWorldTransform(scaler);
	// draw band (artifacts)
	QRect damaged = scalerI.mapRect(ev->rect());
	painter.drawPixmap(damaged, cachedPixmap, damaged);

	// draw current cursor
	QPen pen(seedMode ? Qt::yellow : (*labelColors)[curLabel]);
	pen.setWidth(0);
	painter.setPen(pen);
	painter.drawRect(QRectF(cursor, QSizeF(1, 1)));

	// draw overlay (a quasi one-timer)
	if (overlay) {
		if (scale > 4.) {
			pen.setColor(Qt::yellow); painter.setPen(pen);
			for (int y = 0; y < overlay->rows; ++y) {
				const unsigned char *row = (*overlay)[y];
				for (int x = 0; x < overlay->cols; ++x) {
					if (row[x]) {
						painter.drawLine(x+1, y, x, y+1);
						painter.drawLine(x, y, x+1, y+1);
					}
				}
			}
		} else {
			QImage dest(overlay->cols, overlay->rows, QImage::Format_ARGB32);
			dest.fill(qRgba(0, 0, 0, 0));
			for (int y = 0; y < overlay->rows; ++y) {
				const unsigned char *srcrow = (*overlay)[y];
				QRgb *destrow = (QRgb*)dest.scanLine(y);
				for (int x = 0; x < overlay->cols; ++x) {
					if (srcrow[x])
						destrow[x] = qRgba(255, 255, 0, 191);
				}
			}
			painter.drawImage(0, 0, dest);
		}
	}

	if (seedMode) {
		pen.setColor(Qt::yellow); pen.setWidthF(0.5); pen.setStyle(Qt::DotLine);
		painter.setPen(pen);
		painter.drawRect(0, 0, pixmap->width(), pixmap->height());
	}
}

// helper to color single pixel with labeling
void BandView::updateCachePixel(QPainter &p, int x, int y)
{
	uchar l = labels(y, x);
	if (l > 0) {
		QColor col = (*labelColors)[l];
		col.setAlphaF(0.5);
		p.setPen(col);
		p.drawPoint(x, y);
	}
}

// helper to color single pixel in seed mode
void BandView::updateCachePixelS(QPainter &p, int x, int y)
{
	uchar l = seedMap(y, x);
	if (l == 0 || l == 255) {
		QColor col(l ? Qt::yellow : Qt::red);
		col.setAlphaF(0.5);
		p.setPen(col);
		p.drawPoint(x, y);
	}
}

void BandView::updateCache()
{
	cachedPixmap = pixmap->copy(); // TODO: check for possible qt memory leak
	QPainter painter(&cachedPixmap);

	if (!seedMode) {
		if (showLabels) {
			// mark labeled regions
			for (int y = 0; y < pixmap->height(); ++y)
				for (int x = 0; x < pixmap->width(); ++x)
					updateCachePixel(painter, x, y);
		}
	} else {
		for (int y = 0; y < pixmap->height(); ++y)
			for (int x = 0; x < pixmap->width(); ++x)
				updateCachePixelS(painter, x, y);
	}
	cacheValid = true;
}

void BandView::updateCache(int x, int y)
{
	if (!cacheValid) {
		updateCache();
		return;
	}

	QPixmap &p = cachedPixmap;
	QPainter painter(&p);
	// restore pixel
	painter.drawPixmap(x, y, *pixmap, x, y, 1, 1);

	// if needed, color pixel
	if (!seedMode) {
		if (showLabels)
			updateCachePixel(painter, x, y);
	} else {
		updateCachePixelS(painter, x, y);
	}
}

void BandView::alterLabel(const multi_img::Mask &mask, bool negative)
{
	if (negative) {
		// we are out of luck
		multi_img::MaskConstIt itm = mask.begin();
		multi_img::MaskIt itl = labels.begin();
		for (; itm != mask.end(); ++itm, ++itl)
			if (*itm && (*itl == curLabel))
				*itl = 0;
	} else {
		labels.setTo(curLabel, mask);
	}

	cacheValid = false;
	update();
}

void BandView::drawOverlay(const multi_img::Mask &mask)
{
	//vole::Stopwatch s("Overlay drawing");
	overlay = &mask;
	update();
}

void BandView::cursorAction(QMouseEvent *ev, bool click)
{
	// kill overlay to free the view
	bool grandupdate = (overlay != NULL);

	cursor = QPointF(ev->pos() / scale);
	cursor.setX(round(cursor.x() - 0.75));
	cursor.setY(round(cursor.y() - 0.75));

	// nothing new after all..
	if ((cursor == lastcursor) && !click)
		return;

	int x = cursor.x(), y = cursor.y();

	if (!pixmap->rect().contains(x, y))
		return;

	// overlay in spectral views
	if (grandupdate)
		emit killHover();
	emit pixelOverlay(x, y);

	// paint
	if (ev->buttons() & Qt::LeftButton) {
		if (!seedMode)
			labels(y, x) = curLabel;
		else
			seedMap(y, x) = 0;
		updateCache(x, y);
	// erase
	} else if (ev->buttons() & Qt::RightButton) {
		if (!seedMode) {
			if (labels(y, x) == curLabel) {
				labels(y, x) = 0;
				updateCache(x, y);
				if (!grandupdate)
					updatePoint(cursor);
			}
		} else {
			seedMap(y, x) = 255;
			updateCache(x, y);
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

void BandView::updatePoint(const QPointF &p)
{
	QPoint damagetl = scaler.map(QPoint(p.x() - 2, p.y() - 2));
	QPoint damagebr = scaler.map(QPoint(p.x() + 2, p.y() + 2));
	update(QRect(damagetl, damagebr));
}

void BandView::startGraphseg(const multi_img& input,
							 const vole::GraphSegConfig &config)
{
	vole::GraphSeg seg(config);
	multi_img::Mask result;
	result = seg.execute(input, seedMap);

	/* add segmentation to current labeling */
	multi_img::MaskConstIt sit = result.begin();
	multi_img::MaskIt dit = labels.begin();
	for (; sit < result.end(); ++sit, ++dit)
		if (*sit > 0)
			*dit = curLabel;
		/*else if (*dit == curLabel)
			*dit = 0;*/

	emit seedingDone();
}

void BandView::clearLabelPixels()
{
	for (multi_img::MaskIt it = labels.begin(); it != labels.end(); ++it)
		if (*it == curLabel)
			*it = 0;

	cacheValid = false;
	update();
}

void BandView::leaveEvent(QEvent *ev)
{
	cursor = QPoint(-1, -1);
	update();
}

void BandView::changeLabel(int label)
{
	if (label > -1)
		curLabel = label + 1; // we start with 1, combobox with 0
}

void BandView::toggleSeedMode(bool enabled)
{
	seedMode = enabled;
	cacheValid = false;

	if (enabled) { // starting a new seed
		seedMap = multi_img::Mask(pixmap->height(), pixmap->width(), 127);
	}
	update();
}

void BandView::toggleShowLabels(bool disabled)
{
	if (showLabels == disabled) {	// i.e. not the state we want
		showLabels = !showLabels;
		cacheValid = false;
		update();
	}
}
