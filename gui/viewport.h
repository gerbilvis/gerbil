#ifndef VIEWPORT_H
#define VIEWPORT_H

#include <multi_img.h>
#include <cv.h>
#include <QGLWidget>
#include <vector>
#include <QHash>
#include <QLabel>

struct Bin {
	Bin() {}
	Bin(QVector<QLineF> c, float w) : connections(c), weight(w) {}
	QVector<QLineF> connections;
	float weight;
};

struct BinSet {
	BinSet(const QColor &c) : label(c), totalweight(0.f) {}
	QColor label;
	QHash<QByteArray, Bin> bins;
	float totalweight;
};


class Viewport : public QGLWidget
{
	Q_OBJECT
public:
	Viewport(QWidget *parent = 0);
	void updateModelview();

	int nbins;
	int dimensionality;
	bool gradient;
	std::vector<BinSet> sets;
	std::vector<QString> labels;

	const std::vector<multi_img::Value> *illuminant;

	int selection, hover;
	bool limiterMode;
	std::vector<std::pair<int, int> > limiters;
	bool active, wasActive;

	float useralpha;

	bool showLabeled, showUnlabeled, ignoreLabels;
	bool overlayMode;
	QVector<QLineF> overlayLines;

public slots:
	void killHover();

signals:
	void bandSelected(int dim, bool gradient);
	void newOverlay();
	void activated(bool who);
	void addSelection();
	void remSelection();

protected:
	void paintEvent(QPaintEvent*);
	void resizeEvent(QResizeEvent*);
	void mouseMoveEvent(QMouseEvent*);
	void mousePressEvent(QMouseEvent*);
	void mouseReleaseEvent(QMouseEvent*);
	void wheelEvent(QWheelEvent *);
	void keyPressEvent(QKeyEvent *);

	// helper functions called by mouseMoveEvent
	void updateXY(int sel, int bin);

	// helper functions called by paintEvent
	void drawBins(QPainter&);
	void drawAxesBg(QPainter&);
	void drawAxesFg(QPainter&);
	void drawLegend(QPainter&);
	void drawRegular();
	void drawOverlay();

	// helper for limiter handling
	bool updateLimiter(int dim, int bin);

private:
	// modelview matrix and its inverse
	QTransform modelview, modelviewI;
	// zoom and shift in y-direction
	qreal zoom;
	qreal shift;
	int lasty;

	/* if in limiter mode, user has to release mouse button before switching
	   band. this is for usability, users tend to accidentially switch bands */
	bool holdSelection;

	// cache for efficient overlay
	bool cacheValid;
	QImage cacheImg;
};

#endif // VIEWPORT_H
