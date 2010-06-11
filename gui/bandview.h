#ifndef BANDVIEW_H
#define BANDVIEW_H

#include <QGLWidget>
#include <multi_img.h>

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
	void setSources(const multi_img &i, const multi_img &g);

	multi_img::Mask labels;
	QVector<QColor> markerColors;

public slots:
	void changeLabel(int label);
	void clearLabelPixels();
	void alterLabel(const multi_img::Mask &mask, bool negative);
	void drawOverlay(const multi_img::Mask &mask);

	void toggleSeedMode(bool enabled);
	void startGraphseg();

signals:
	void seedingDone(bool yeah = false);

private:
	void cursorAction(QMouseEvent *ev, bool click = false);
	inline void updateCachePixel(QPainter &p, int x, int y);
	inline void updateCachePixelS(QPainter &p, int x, int y);
	void updateCache();
	void updateCache(int x, int y);
	void updatePoint(const QPointF &p);

	qreal scale;
	QTransform scaler, scalerI;
	QPointF cursor, lastcursor;
	const QPixmap *pixmap;
	QPixmap cachedPixmap;
	bool cacheValid;

	uchar curLabel;
	const multi_img::Mask *overlay;

	/// interpret input as segmentation seeds
	bool seedMode;
	multi_img::Mask seedMap;
	const multi_img *sources[2];
};

#endif // BANDVIEW_H
