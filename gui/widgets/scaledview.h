#ifndef SCALEDVIEW_H
#define SCALEDVIEW_H

#include <QGraphicsScene>
#include <QPainter>

class QGLWidget;

class ScaledView : public QGraphicsScene
{
	Q_OBJECT
public:
	ScaledView();
	virtual ~ScaledView() {}

	// handles both resize and drawing
	void drawBackground(QPainter *painter, const QRectF &rect);
	virtual void resizeEvent();
	virtual void paintEvent(QPainter *painter, const QRectF &rect);
	void mouseMoveEvent(QGraphicsSceneMouseEvent*);
	void mousePressEvent(QGraphicsSceneMouseEvent*);

	virtual void setPixmap(QPixmap p);

	/* provide a reasonably high size of correct aspect ratio for layouting */
	virtual QSize updateSizeHint();

signals:
	void newSizeHint(QSize hint);

protected:
	virtual void cursorAction(QGraphicsSceneMouseEvent *ev,
							  bool click = false);
	void drawWaitMessage(QPainter *painter);

	// draw the background, inline function
	void fillBackground(QPainter *painter, const QRectF& rect) {
		static QBrush brush(QColor(63, 31, 63), Qt::Dense4Pattern);
		painter->fillRect(rect, brush);
	}

	int width, height;

	qreal scale;
	QTransform scaler, scalerI;
	QPixmap	pixmap;
};

#endif // SCALEDVIEW_H