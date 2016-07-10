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
#include <unordered_map>
#include <utility>

class ModeWidget;

class BandView : public ScaledView
{
	Q_OBJECT
public:

	enum CursorSize
	{
		Small,
		Medium,
		Big,
		Huge
	};

	enum class CursorMode
	{
		Marker,
		Rubber
	};

	enum class OverrideMode
	{
		On,
		Off
	};

	BandView();

	void initUi();

	void setPixmap(QPixmap pixmap);
	void setLabelMatrix(const cv::Mat1b & matrix);

	int getCurrentLabel() { return curLabel; }
	cv::Mat1s getSeedMap() { return seedMap; }
	void setSeedMap(cv::Mat1s seeding);

	void setZoomAction(QAction* act) { zoomAction = act; }
	void setLabelAction(QAction* act) { labelAction = act; }
	void setPickAction(QAction* act) { pickAction = act; }

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
	void toggleSeedMode(bool enabled);
	void clearSeeds();
	void toggleLabelHighlight(short label);

	void enterEvent();
	void leaveEvent();

	void updateCursorSize(BandView::CursorSize size);
	void toggleCursorMode();
	void toggleOverrideMode();

	void updatePixel(int x, int y);

protected slots:
	void saveState();

signals:
	void killHover();

	// picking mode, diff. label chosen
	void labelSelected(int label);

	// user changed some labels
	void alteredLabels(const cv::Mat1s &labels, const cv::Mat1b &mask);

	// user wants full labeling update
	void newLabeling(const cv::Mat1s &labels);

	// user requested additional label
	void newLabel();

	// user wants to clear a label
	void clearRequested();

	void mergeLabelsRequested(QVector<int> labels);

	void setAlphaValue(int val);

protected:
	void paintEvent(QPainter *painter, const QRectF &rect);
	void keyPressEvent(QKeyEvent *);
	QMenu* createContextMenu();
	void restoreState();

private:
	void cursorAction(QGraphicsSceneMouseEvent *ev, bool click = false);
	inline void markCachePixel(QPainter &p, int x, int y);
	inline void markCachePixelS(QPainter &p, int x, int y);
	void updateCache();
	void updateCache(int y, int x, short label = 0);
	void updatePoint(const QPoint &p);
	static std::pair<QPolygonF, QPolygonF>
	createCursor(const cv::Mat1b &mask, const QPoint &center);
	void initCursors();

	QPolygonF getCursor(int xpos, int ypos);
	QPolygonF getCursorHull(int xpos, int ypos);

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
	QVector<int> selectedLabels;
	const cv::Mat1b *overlay;

	/// color view according to labels
	bool showLabels;

	/// input mode to return to after seeding
	InputMode lastMode;
	
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

	CursorMode cursorMode = CursorMode::Marker;
	CursorSize cursorSize = CursorSize::Medium;
	OverrideMode overrideMode = OverrideMode::On;

	// point sets, stored as polygon for translate(), and hulls for drawing
	std::unordered_map<CursorSize, std::pair<QPolygonF, QPolygonF>,
	std::hash<int>> cursors;

	QAction* zoomAction = nullptr;
	QAction* labelAction = nullptr;
	QAction* pickAction = nullptr;
};

#endif // BANDVIEW_H
