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

	bool showLabeled, showUnlabeled;
signals:
	void sliceSelected(int dim, bool gradient);

protected:
	void paintEvent(QPaintEvent *event);
	void mouseMoveEvent(QMouseEvent *event);

private:

	int selection, hover;
};

class SliceView : public QLabel
{
	Q_OBJECT
public:
	SliceView(QWidget *parent = 0);
	void resizeEvent(QResizeEvent * ev);
	void paintEvent(QPaintEvent *ev);
	void mouseMoveEvent(QMouseEvent *ev);
	void leaveEvent(QEvent *ev);
	void setPixmap(const QPixmap &);

	QImage *labels;
	QVector<QColor> markerColors;

public slots:
	void changeLabel(int label);
	void clearLabelPixels();

private:
	void updateCache();
	void updatePoint(const QPointF &p);

	qreal scale;
	QTransform scaler, scalerI;
	QPointF cursor, lastcursor;
	QPixmap cachedPixmap;
	bool cacheValid;

	int curLabel;
};

#endif // VIEWPORT_H
