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

	void setPixmap(const QPixmap &pixmap);

	QTimer labelTimer;
	cv::Mat1s labels;
	cv::Mat1s seedMap; // mat1s to be consistent with labels matrix

public slots:
	void refresh();
	void changeLabel(int label);
	void clearLabelPixels();
	void clearAllLabels(); /// TODO: add a button for this
	void alterLabel(const multi_img::Mask &mask, bool negative);
	void setLabels(multi_img::Mask l);
	void updateLabels();
	void commitLabelChanges();
	void drawOverlay(const multi_img::Mask &mask);

	void setLabelColors(const QVector<QColor> &labelColors, bool changed);
	void applyLabelAlpha(int alpha);
	void toggleShowLabels(bool disabled);
	void toggleSeedMode(bool enabled);

signals:
	void pixelOverlay(int x, int y);
	void subPixels(const std::map<std::pair<int, int>, short> &points);
	void addPixels(const std::map<std::pair<int, int>, short> &points);
	void refreshLabels();
	void killHover();
	void newLabel(); // user requested another label

private:
	void cursorAction(QMouseEvent *ev, bool click = false);
	inline void markCachePixel(QPainter &p, int x, int y);
	inline void markCachePixelS(QPainter &p, int x, int y);
	void updateCache();
	void updateCache(int x, int y, short label = 0);
	void updatePoint(const QPointF &p);

	std::map<std::pair<int, int>, short> uncommitedLabels;

	QPixmap cachedPixmap;
	bool cacheValid;

	QPointF cursor, lastcursor;
	short curLabel;
	const multi_img::Mask *overlay;

	/// color view according to labels
	bool showLabels;

	/// interpret input as segmentation seeds
	bool seedMode;
	
	int labelAlpha;
	QVector<QColor> labelColors;
	QVector<QColor> labelColorsA; /// labelColors with user-selected alpha
	std::pair<QColor, QColor> seedColorsA;
};

#endif // BANDVIEW_H
