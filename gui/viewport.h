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

	bool active;
	int selection, hover;
	float useralpha;

signals:
	void bandSelected(int dim, bool gradient);
	void newOverlay();
	void activated(bool who);

protected:
	void paintEvent(QPaintEvent*);
	void resizeEvent(QResizeEvent*);
	void mouseMoveEvent(QMouseEvent*);
	void mousePressEvent(QMouseEvent*);
	void mouseReleaseEvent(QMouseEvent*);
	void wheelEvent(QWheelEvent *);

	// helper functions called by paintEvent
	void paintBins(QPainter&);
	void paintAxes(QPainter&, bool fore);
	void paintLegend(QPainter&);

private:
	// modelview matrix and its inverse
	QTransform modelview, modelviewI;
	// zoom and shift in y-direction
	qreal zoom;
	qreal shift;
	int lasty;
};

#endif // VIEWPORT_H
