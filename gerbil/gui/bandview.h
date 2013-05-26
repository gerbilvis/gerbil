/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef BANDVIEW_H
#define BANDVIEW_H

#include "scaledview.h"

#include <multi_img.h>
#include <map>
#include <QPen>
#include <QTimer>

class BandView : public ScaledView
{
	Q_OBJECT
public:
	BandView(QWidget *parent = 0);
	void paintEvent(QPaintEvent *ev);
	void leaveEvent(QEvent *ev);

	void setPixmap(QPixmap pixmap);
	void setLabelMatrix(cv::Mat1s matrix);

	bool isSeedModeEnabled() { return seedMode; }

	short getCurLabel() { return curLabel; }

	// TODO: these are accessed by MainWindow
	QTimer labelTimer;
	cv::Mat1s seedMap; // mat1s to be consistent with labels matrix
	cv::Mat1b curMask; // in single label mode contains curlabel members

public slots:
	void refresh();
	void changeCurrentLabel(int label);
	void setLabels(cv::Mat1b l);
	void updateLabels();
	void commitLabelChanges();
	void drawOverlay(const cv::Mat1b &mask);

	void updateLabeling(const QVector<QColor> &labelColors, bool changed);
	void applyLabelAlpha(int alpha);
	void toggleShowLabels(bool disabled);
	void toggleSingleLabel(bool enabled);
	void toggleSeedMode(bool enabled);
	void clearSeeds();

signals:
	void pixelOverlay(int x, int y);
	void subPixels(const std::map<std::pair<int, int>, short> &points);
	void addPixels(const std::map<std::pair<int, int>, short> &points);

	void refreshLabels();
	void killHover();

	// user requested additional label
	void newLabel();
	// single label mode, diff. label chosen
	void newSingleLabel(short label);

private:
	void cursorAction(QMouseEvent *ev, bool click = false);
	inline void markCachePixel(QPainter &p, int x, int y);
	inline void markCachePixelS(QPainter &p, int x, int y);
	void updateCache();
	void updateCache(int x, int y, short label = 0);
	void updatePoint(const QPointF &p);

	// local reference to global labeling matrix
	cv::Mat1s labels;

	// pixel labels we did change, but not yet notify other parties about
	std::map<std::pair<int, int>, short> uncommitedLabels;

	QPixmap cachedPixmap;
	bool cacheValid;

	QPointF cursor, lastcursor;
	short curLabel;
	const cv::Mat1b *overlay;

	/// color view according to labels
	bool showLabels, singleLabel, holdLabel;

	/// interpret input as segmentation seeds
	bool seedMode;
	
	/// local copy of label colors
	QVector<QColor> labelColors;
	/// user-selected alpha
	int labelAlpha;
	/// labelColors with user-selected alpha
	QVector<QColor> labelColorsA;
	std::pair<QColor, QColor> seedColorsA;
};

#endif // BANDVIEW_H
