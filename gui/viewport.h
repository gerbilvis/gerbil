#ifndef VIEWPORT_H
#define VIEWPORT_H

#include <QGLWidget>
#include <vector>
#include <iostream>
#include <QHash>
#include <QLabel>
#include <cv.h>

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

	bool showLabeled, showUnlabeled, ignoreLabels;

	std::vector<float> *illuminant;

	bool active;
	int selection, hover;
	float useralpha;

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

	// helper functions called by paintEvent
	void drawBins(QPainter&);
	void drawAxes(QPainter&, bool fore);
	void drawLegend(QPainter&);
	void drawRegular();
	void drawOverlay();

private:
	// modelview matrix and its inverse
	QTransform modelview, modelviewI;
	// zoom and shift in y-direction
	qreal zoom;
	qreal shift;
	int lasty;

	// cache for efficient overlay
	bool cacheValid;
	QImage cacheImg;
};

#endif // VIEWPORT_H
