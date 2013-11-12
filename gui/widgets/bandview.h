/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef BANDVIEW_H
#define BANDVIEW_H

#include "widgets/scaledview.h"

#include <multi_img.h>
#include <map>
#include <QPen>
#include <QTimer>

class BandView : public ScaledView
{
	Q_OBJECT
public:
	BandView();

	void initUi();

	void setPixmap(QPixmap pixmap);
	void setLabelMatrix(const cv::Mat1b & matrix);

	bool isSeedModeEnabled() { return seedMode; }

	int getCurrentLabel() { return curLabel; }
	cv::Mat1s getSeedMap() { return seedMap; }
	void setSeedMap(cv::Mat1s seeding);

public slots:
	void refresh();
	void commitLabels();
	void commitLabelChanges();
	void drawOverlay(const cv::Mat1b &mask);

	void setCurrentLabel(int label);
	/* update either labeling colors, or both them and pixel labels */
	void updateLabeling(const cv::Mat1s &labels,
						const QVector<QColor> &colors = QVector<QColor>(),
						bool colorsChanged = false);
	void updateLabeling(const cv::Mat1s &labels, const cv::Mat1b &mask);
	void applyLabelAlpha(int alpha);
	void toggleShowLabels(bool disabled);
	void toggleSingleLabel(bool enabled);
	void toggleSeedMode(bool enabled);
	void clearSeeds();
	void highlightSingleLabel(short label, bool highlight);

	void enterEvent();
	void leaveEvent();

signals:
	void pixelOverlay(int y, int x);
	void killHover();

	// single label mode, diff. label chosen
	void singleLabelSelected(int label);

	// user changed some labels
	void alteredLabels(const cv::Mat1s &labels, const cv::Mat1b &mask);

	// user wants full labeling update
	void newLabeling(const cv::Mat1s &labels);

	// user requested additional label
	void newLabel();

	// user wants to clear a label
	void clearRequested();

protected:
	void paintEvent(QPainter *painter, const QRectF &rect);
	void keyPressEvent(QKeyEvent *);

private:
	void cursorAction(QGraphicsSceneMouseEvent *ev, bool click = false);
	inline void markCachePixel(QPainter &p, int x, int y);
	inline void markCachePixelS(QPainter &p, int x, int y);
	void updateCache();
	void updateCache(int y, int x, short label = 0);
	void updatePoint(const QPoint &p);

	// local labeling matrix
	cv::Mat1s labels;

	// mask that contains pixel labels we did change, but not commit back yet
	cv::Mat1b uncommitedLabels;

	// ignore the signals when we were the originator
	bool ignoreUpdates;

	// the cachedPixmap is colored with label colors
	QPixmap cachedPixmap;
	bool cacheValid;

	QPoint cursor, lastcursor;
	short curLabel;
	const cv::Mat1b *overlay;

	/// color view according to labels
	bool showLabels, singleLabel, holdLabel;

	/// interpret input as segmentation seeds
	bool seedMode;
	
	// local copy of label colors
	QVector<QColor> labelColors;
	/// user-selected alpha
	int labelAlpha;
	/// labelColors with user-selected alpha
	QVector<QColor> labelColorsA;
	std::pair<QColor, QColor> seedColors;

	QTimer labelTimer;
	cv::Mat1s seedMap; // mat1s to be consistent with labels matrix
	cv::Mat1b curMask; // in single label mode contains curlabel members
};

#endif // BANDVIEW_H
