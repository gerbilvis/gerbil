/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VIEWPORT_H
#define VIEWPORT_H

#include "distviewcompute.h"

#include <QGraphicsScene>
#include <QGLWidget>
#include <QLabel>
#include <QTimer>

#include <vector>

class Viewport : public QGraphicsScene
{
	Q_OBJECT
public:
	Viewport(representation::t type, QGLWidget *target);
	~Viewport();

	void prepareLines();
	void setLimiters(int label);

	enum RenderMode {
		RM_SKIP = 0,
		RM_STEP = 1,
		RM_FULL = 2
	};

	enum class BufferFormat : GLenum {
		RGBA8 = GL_RGBA8,//0x8058,  // GL_RGBA8, constants not defined on windows
		RGBA16F = GL_RGBA16F,//0x881A,// GL_RGBA16F
		RGBA32F = GL_RGBA32F//0x8814 // GL_RGBA32F
	};

	BufferFormat format() { return bufferFormat; }

	/* TODO: make non-public. I am just too tired right now. */
	// viewport context
	vpctx_ptr ctx;
	// histograms (binsets)
	sets_ptr sets;
	bool active;

public slots:

	void toggleLabelHighlight(int index);

	void setAlpha(float alpha);

	void setLimitersMode(bool enabled);

	/** Application focus on this viewport. */
	void activate();

	// entry point of user interaction with temporary quick drawing
	void startNoHQ();
	// exit point of temporary quick drawing. returns true if redraw performed
	bool endNoHQ(RenderMode spectrum = RM_STEP, RenderMode highlight = RM_STEP);

	void screenshot();

	void rebuild();

	// toggle which sets are shown
	void toggleLabeled(bool enabled);
	void toggleUnlabeled(bool enabled);

	// pixel overlay
	void removePixelOverlay();
	void insertPixelOverlay(const QPolygonF &points);

	// illuminant correction
	void changeIlluminantCurve(QVector<multi_img::Value> illum);
	void setIlluminationCurveShown(bool show);
	void setAppliedIlluminant(QVector<multi_img::Value> illum);

	void setBufferFormat(BufferFormat format, bool propagate = false);
	void toggleBufferFormat();
	void toggleHQ();

	void setDrawLog(QAction* logAct) { drawLog = logAct; }
	void setDrawHQ(QAction* hqAct) { drawHQ = hqAct; }
	void setDrawRGB(QAction* rgbAct) { drawRGB = rgbAct; }
	void setDrawMeans(QAction* meansAct) { drawMeans = meansAct; }
	void restoreState();

protected slots:

	// triggered by renderTimers
	void continueDrawing(int buffer);

	// triggered by resizeTimer
	void resizeScene();

	// triggered manually
	void updateBuffers(RenderMode spectrum = RM_STEP,
	                   RenderMode highlight = RM_STEP);

	void saveState();

signals:
	// we are the active viewer
	void activated(representation::t type);

	// selection changed -> band display
	void bandSelected(int dim);
	// highlight changed -> band display overlay
	void requestOverlay(int dim, int bin);
	void requestOverlay(const std::vector<std::pair<int, int> >& limiters,
	                    int dim);

	// add/remove highlight from/to current label
	void addSelectionRequested();
	void remSelectionRequested();

	void bufferFormatToggled(Viewport::BufferFormat format);

protected:
	void initTimers();
	// called on resize
	void initBuffers();
	bool tryInitBuffers();

	void reset();
	// handles both resize and drawing
	void drawBackground(QPainter *painter, const QRectF &rect);

	void mouseMoveEvent(QGraphicsSceneMouseEvent*);
	void mousePressEvent(QGraphicsSceneMouseEvent*);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent*);
	void wheelEvent(QGraphicsSceneWheelEvent *);
	void keyPressEvent(QKeyEvent *);

	//helper function that adjusts boundaries after zooming/panning
	void adjustBoundaries();

	// helper function that updates Y-axis labels
	void updateYAxis(bool yAxisChanged = false);

	// helper function that updates world transformation
	void updateModelview(bool newBinning = false);

	// helper functions called by mouseMoveEvent
	bool updateXY(int sel, int bin);

	// draw the full scene (returns false if not everything could be drawn)
	bool drawScene(QPainter*, bool withDynamics = true);

	/* helper functions called by drawScene/updateTextures */

	void drawBins(QPainter &painter, QTimer &renderTimer,
	              unsigned int &renderedLines, unsigned int renderStep,
	              bool onlyHighlight);
	// helper function called by drawBins
	QColor determineColor(const QColor &basecolor, float weight,
	                      float totalweight, bool highlighted, bool single);

	void drawAxesBg(QPainter*);
	void drawAxesFg(QPainter*);
	// parameter selection for highlighted band (in red), omit if not desired
	void drawLegend(QPainter*, int selection = -1);
	void drawOverlay(QPainter*);
	void drawWaitMessage(QPainter*);

	// helper for limiter handling
	bool updateLimiter(int dim, int bin);

private:
	representation::t type;
	int width, height;

	struct renderbuffer {
		renderbuffer() : fbo(0), blit(0), dirty(true),
		    renderStep(10000), renderedLines(0) {}

		// buffer to render to
		QGLFramebufferObject *fbo;
		// buffer for blitting
		QGLFramebufferObject *blit;
		// true: fbo is not initialized, or was not drawn-to yet.
		bool dirty;
		// how many elements to render per step
		const unsigned int renderStep;
		// how many elements were already rendered
		unsigned int renderedLines;
		// timer for incremental rendering
		QTimer renderTimer;
	};

	renderbuffer buffers[2];

	// normalized illuminant spectrum for drawing the curve
	QVector<multi_img::Value> illuminantCurve;
	// draw the illuminant curve
	bool illuminant_show;
	// draw vectors skewed according to illuminant
	std::vector<multi_img::Value> illuminantAppl;

	int selection, hover;
	bool limiterMode;
	std::vector<std::pair<int, int> > limiters;

	float useralpha;

	bool showLabeled, showUnlabeled;
	bool overlayMode;
	QPolygonF overlayPoints;

	// target widget needed for GL context
	QGLWidget *target;

	// vertex buffer
	QGLBuffer vb;
	// index to vertex buffer
	binindex shuffleIdx;

	// modelview matrix and its inverse
	QTransform modelview, modelviewI;

	qreal zoom;

	/* if in limiter mode, user has to release mouse button before switching
	   band. this is for usability, users tend to accidentially switch bands */
	bool holdSelection;
	int *activeLimiter;

	// draw with log weights vs. linear weights
	QAction* drawLog;
	// drawing mode mean vs. bin center
	QAction* drawMeans;
	// drawing mode labelcolor vs. sRGB color
	QAction* drawRGB;
	// draw with antialiasing
	QAction* drawHQ;
	// texture buffer format
	BufferFormat bufferFormat;

	enum {
		HIGH_QUALITY,        // drawing HQ as usual
		HIGH_QUALITY_QUICK,  // last update was HQ, quick updates requested
		QUICK                // last update not HQ, quick updates requested
	} drawingState;

	// this timer will resize the scene after resize/folding
	QTimer resizeTimer;

	std::vector<QString> yaxis;
	int yaxisWidth;
	int displayHeight; // height of plot without paddings

	// vector containing highlighted labels
	QVector<int> highlightLabels;

	struct {
		int hp = 20; //horizontal padding
		int vp = 12; //vertical padding
		int vtp = 22; // lower padding for text (legend)
		int htp; // left padding for text (legend)
	} boundaries;
};

Q_DECLARE_METATYPE(Viewport::BufferFormat)

#endif // VIEWPORT_H
