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

	bool showLabeled, showUnlabeled;

	bool active;
	int selection, hover;

signals:
	void sliceSelected(int dim, bool gradient);
	void newOverlay();
	void activated(bool who);

protected:
	void paintEvent(QPaintEvent*);
	void mouseMoveEvent(QMouseEvent*);
	void mousePressEvent(QMouseEvent*);

private:
	QTransform modelviewI;
};

class SliceView : public QGLWidget
{
	Q_OBJECT
public:
	SliceView(QWidget *parent = 0);
	void resizeEvent(QResizeEvent * ev);
	void paintEvent(QPaintEvent *ev);
	void mouseMoveEvent(QMouseEvent *ev) { cursorAction(ev); }
	void mousePressEvent(QMouseEvent *ev) { cursorAction(ev, true); }
	void leaveEvent(QEvent *ev);
	void setPixmap(const QPixmap &);

	cv::Mat_<uchar> labels;
	QVector<QColor> markerColors;

public slots:
	void changeLabel(int label);
	void clearLabelPixels();
	void alterLabel(const cv::Mat_<uchar> &mask, bool negative);
	void drawOverlay(const cv::Mat_<uchar> &mask);

private:
	void cursorAction(QMouseEvent *ev, bool click = false);
	void updateCache();
	void updatePoint(const QPointF &p);

	qreal scale;
	QTransform scaler, scalerI;
	QPointF cursor, lastcursor;
	const QPixmap *pixmap;
	QPixmap cachedPixmap;
	bool cacheValid;

	uchar curLabel;
	const cv::Mat_<uchar> *overlay;
};

#endif // VIEWPORT_H
