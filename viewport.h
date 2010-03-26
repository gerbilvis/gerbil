#ifndef VIEWPORT_H
#define VIEWPORT_H

#include <QGLWidget>
#include <vector>
#include <iostream>
#include <QHash>
#include <QLabel>

struct Bin {
	Bin() {}
	Bin(QVector<QLineF> c, float w) : connections(c), weight(w) {}
	QVector<QLineF> connections;
	float weight;
};

struct BinSet {
	QColor label;
	QHash<QByteArray, Bin> bins;
	float totalweight;
};


class Viewport : public QGLWidget
{
	Q_OBJECT
public:
	Viewport(QWidget *parent = 0);
	void addSet(const BinSet *set)	{ sets.push_back(set); }
	QTransform getModelview();

	int nbins;
	int dimensionality;
	bool gradient;

signals:
	void sliceSelected(int dim);

protected:
	void paintEvent(QPaintEvent *event);
	void mouseMoveEvent(QMouseEvent *event);

private:
	std::vector<const BinSet*> sets;

	int selection, hover;
};

class SliceLabel : public QLabel
{
	Q_OBJECT
public:
	SliceLabel(QWidget *parent = 0);
	void paintEvent(QPaintEvent *event);
};

#endif // VIEWPORT_H
