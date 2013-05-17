/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VIEWPORT_H
#define VIEWPORT_H

#include "compute.h"
#include "viewportcontrol.h"
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGLWidget>
#include <vector>
#include <QLabel>
#include <QTimer>
#include <QResizeEvent>

class Viewport : public QGraphicsScene
{
	Q_OBJECT
public:
	Viewport(ViewportControl *control, QGLWidget *target);
	~Viewport();

	void prepareLines();
	void setLimiters(int label);

	bool illuminant_correction;
	std::vector<multi_img::Value> illuminant;

	int selection, hover;
	bool limiterMode;
	std::vector<std::pair<int, int> > limiters;
	bool active, wasActive;

	float useralpha;

	bool showLabeled, showUnlabeled;
	bool overlayMode;
	QPolygonF overlayPoints;

	enum RenderMode {
		RM_SKIP = 0,
		RM_STEP = 1,
		RM_FULL = 2
	};

	// we would like this not to be public (TODO), viewportcontrol accesses it
	QGLWidget *target;
	ViewportControl *control;
	QGraphicsItem *controlItem;

	// viewport context
	vpctx_ptr ctx;
	// histograms (binsets)
	sets_ptr sets;

	QGLBuffer vb;
	binindex shuffleIdx;

public slots:
	void killHover();
	void highlight(short index);
	void toggleRGB(bool enabled)
	{ drawRGB = enabled; updateTextures(); }
	void activate();

	// entry and exit point of user interaction with quick drawing
	void startNoHQ(bool resize = false);
	bool endNoHQ();
	void resizeEpilog();

	// acknowledge folding
	void folding() { drawingState = FOLDING; resizeTimer.start(50); }

	void screenshot();

	void rebuild();

	void updateTextures(RenderMode spectrum = RM_STEP, RenderMode highlight = RM_STEP);

protected slots:
	void continueDrawingSpectrum();
	void continueDrawingHighlight();

signals:
	void bandSelected(representation type, int dim);
	void newOverlay(int dim);
	void activated();
	void addSelection();
	void remSelection();

protected:
	void reset();
	// handles both resize and drawing
	void drawBackground(QPainter *painter, const QRectF &rect);

	void resizeEvent();
	void mouseMoveEvent(QGraphicsSceneMouseEvent*);
	void mousePressEvent(QGraphicsSceneMouseEvent*);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent*);
	void wheelEvent(QGraphicsSceneWheelEvent *);
	void keyPressEvent(QKeyEvent *);

	// helper function that updates Y-axis labels
	void updateYAxis();

	// helper function that updates world transformation
	void updateModelview();

	// helper functions called by mouseMoveEvent
	void updateXY(int sel, int bin);

	// helper functions called by drawBackground
	void drawBins(QPainter &painter, QTimer &renderTimer,
		unsigned int &renderedLines, unsigned int renderStep, bool onlyHighlight);
	// helper function called by drawBins
	QColor determineColor(const QColor &basecolor, float weight,
						  float totalweight, bool highlighted, bool single);

	void drawAxesBg(QPainter*);
	void drawAxesFg(QPainter*);
	void drawLegend(QPainter*);
	void drawOverlay(QPainter*);
	void drawWaitMessage(QPainter*);

	// helper for limiter handling
	bool updateLimiter(int dim, int bin);

private:
	int width, height;
	QGLFramebufferObject *fboSpectrum;
	QGLFramebufferObject *fboHighlight;
	QGLFramebufferObject *fboMultisamplingBlit;

	// modelview matrix and its inverse
	QTransform modelview, modelviewI;
	// zoom and shift in y-direction
	qreal zoom;
	qreal shift;
	int lasty;

	/* if in limiter mode, user has to release mouse button before switching
	   band. this is for usability, users tend to accidentially switch bands */
	bool holdSelection;
	int *activeLimiter;

	// cache for efficient overlay
	bool cacheValid;
	QImage cacheImg;

	// draw without highlight
	bool clearView;
	bool implicitClearView;

	// drawing mode mean vs. bin center
	bool drawMeans;
	// drawing mode labelcolor vs. sRGB color
	bool drawRGB;
	// draw with antialiasing
	bool drawHQ;

	enum {
		HIGH_QUALITY,        // drawing HQ as usual
		HIGH_QUALITY_QUICK,  // last update was HQ, quick updates requested
		QUICK,               // last update not HQ, quick updates requested
		RESIZE,              // resize updates requested (last update invalid)
		SCREENSHOT,          // screenshot update requested (special drawing)
		FOLDING              // only draw blank during folding resize ops
	} drawingState;
	// this timer will re-enable regular drawing after resize/folding
	QTimer resizeTimer;
	QTimer scrollTimer;

	static const unsigned int renderAtOnceStep;

	static const unsigned int spectrumRenderStep;
	QTimer spectrumRenderTimer;
	unsigned int spectrumRenderedLines;

	static const unsigned int highlightRenderStep;
	QTimer  highlightRenderTimer;
	unsigned int  highlightRenderedLines;

	std::vector<QString> yaxis;
	int yaxisWidth;

	// single label to be highlighted
	int highlightLabel;
};

class ViewportGV : public QGraphicsView
{
public:
	ViewportGV(QWidget *parent) : QGraphicsView(parent)
	{}

protected:
	void resizeEvent(QResizeEvent *event) {
		if (scene())
			scene()->setSceneRect(QRect(QPoint(0, 0), event->size()));
		QGraphicsView::resizeEvent(event);
	}
};

#endif // VIEWPORT_H
