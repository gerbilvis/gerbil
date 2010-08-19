#ifndef BANDVIEW_H
#define BANDVIEW_H

#include <multi_img.h>
#include <graphseg.h>

#include <QGLWidget>

class BandView : public QGLWidget
{
	Q_OBJECT
public:
	BandView(QWidget *parent = 0);
	void resizeEvent(QResizeEvent * ev);
	void paintEvent(QPaintEvent *ev);
	void mouseMoveEvent(QMouseEvent *ev) { cursorAction(ev); }
	void mousePressEvent(QMouseEvent *ev) { cursorAction(ev, true); }
	void leaveEvent(QEvent *ev);
	void setPixmap(const QPixmap &pixmap);

	multi_img::Mask labels;
	QVector<QColor> *labelColors;

public slots:
	void changeLabel(int label);
	void clearLabelPixels();
	void alterLabel(const multi_img::Mask &mask, bool negative);
	void drawOverlay(const multi_img::Mask &mask);

	void toggleShowLabels(bool disabled);
	void toggleSeedMode(bool enabled);
	void startGraphseg(const multi_img& input, const vole::GraphSegConfig &config);

signals:
	void seedingDone(bool yeah = false);
	void pixelOverlay(int x, int y);
	void killHover();

private:
	void cursorAction(QMouseEvent *ev, bool click = false);
	inline void updateCachePixel(QPainter &p, int x, int y);
	inline void updateCachePixelS(QPainter &p, int x, int y);
	void updateCache();
	void updateCache(int x, int y);
	void updatePoint(const QPointF &p);

	qreal scale;
	QTransform scaler, scalerI;
	const QPixmap *pixmap;
	QPixmap cachedPixmap;
	bool cacheValid;

	QPointF cursor, lastcursor;
	uchar curLabel;
	const multi_img::Mask *overlay;

	/// color view according to labels
	bool showLabels;

	/// interpret input as segmentation seeds
	bool seedMode;
	multi_img::Mask seedMap;
};

#endif // BANDVIEW_H
