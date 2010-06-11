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
	QTransform getModelview();

	int nbins;
	int dimensionality;
	bool gradient;
	std::vector<BinSet> sets;

	bool showLabeled, showUnlabeled, ignoreLabels;

	bool active;
	int selection, hover;

signals:
	void bandSelected(int dim, bool gradient);
	void newOverlay();
	void activated(bool who);

protected:
	void paintEvent(QPaintEvent*);
	void mouseMoveEvent(QMouseEvent*);
	void mousePressEvent(QMouseEvent*);
	void mouseReleaseEvent(QMouseEvent*);
	void wheelEvent(QWheelEvent *);

private:
	QTransform modelviewI;
	// zoom and shift in y-direction
	qreal zoom;
	qreal shift;
	int lasty;
};

#endif // VIEWPORT_H
