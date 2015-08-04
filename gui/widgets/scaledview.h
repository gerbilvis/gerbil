#ifndef SCALEDVIEW_H
#define SCALEDVIEW_H

#include <QGraphicsScene>
#include <QPainter>

class QGLWidget;

class ScaledView : public QGraphicsScene
{
	Q_OBJECT
public:
	enum class InputMode
	{
		Zoom,
		Pick,
		Label,
		Seed,
		Disabled
	};

	ScaledView();
	virtual ~ScaledView() {}

	virtual void setPixmap(QPixmap p);

	/* provide a reasonably high size of correct aspect ratio for layouting */
	virtual void updateSizeHint();

	// offsets reserved for autohidewidgets, can be altered from outside
	int offLeft, offTop, offRight, offBottom;

signals:
	void newSizeHint(QSize hint);
	void newContentRect(QRect rect);

protected:
	// handles both resize and drawing
	void drawBackground(QPainter *painter, const QRectF &rect);
	virtual void resizeEvent();
	virtual void paintEvent(QPainter *painter, const QRectF &rect);
	void mouseMoveEvent(QGraphicsSceneMouseEvent*);
	void mousePressEvent(QGraphicsSceneMouseEvent*);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent*);
	virtual void cursorAction(QGraphicsSceneMouseEvent *ev,
							  bool click = false);
	void wheelEvent(QGraphicsSceneWheelEvent*);

	void drawWaitMessage(QPainter *painter);

	// draw the background, inline function
	void fillBackground(QPainter *painter, const QRectF& rect) {
		static QBrush brush(Qt::gray, Qt::Dense4Pattern);
		painter->fillRect(rect, Qt::black);
		painter->fillRect(rect, brush);
	}

	// always call after changes to scaler
	void scalerUpdate();

	// scene geometry
	int width, height;
	qreal zoom;
	InputMode inputMode;

	// transformations between pixmap coords. and scene coords.
	QTransform scaler, scalerI;

	// the pixmap we display
	QPixmap	pixmap;
};

#endif // SCALEDVIEW_H
