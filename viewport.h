#ifndef VIEWPORT_H
#define VIEWPORT_H

#include <QGLWidget>
#include <vector>
#include <iostream>
#include <QHash>


struct Bin {
	Bin() {}
	Bin(QVector<QLineF> c, float w) : connections(c), weight(w) {}
	QVector<QLineF> connections;
	float weight;
};

struct BinSet {
	QColor label;
	QHash<qlonglong, Bin> bins;
	float totalweight;
};


class Viewport : public QGLWidget
{
	Q_OBJECT
public:
	Viewport(QWidget *parent = 0);
	void addSet(const BinSet *set)	{ sets.push_back(set); }

	int nbins;
	int dimensionality;
protected:
	void paintEvent(QPaintEvent *event);

private:
	std::vector<const BinSet*> sets;

};

#endif // VIEWPORT_H
