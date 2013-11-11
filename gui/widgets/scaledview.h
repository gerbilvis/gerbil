#ifndef SCALEDVIEW_H
#define SCALEDVIEW_H

#include <QGLWidget>

class ScaledView : public QGLWidget
{
	Q_OBJECT
public:
	ScaledView(QWidget *parent = 0);
	virtual ~ScaledView() {}
	void resizeEvent(QResizeEvent * ev);
	virtual void paintEvent(QPaintEvent *ev);
	void mouseMoveEvent(QMouseEvent *ev);
	void mousePressEvent(QMouseEvent *ev);
	virtual void setPixmap(QPixmap p);

	/* provide a reasonably high size of correct aspect ratio for layouting */
	virtual QSize sizeHint() const;

protected:
	virtual void cursorAction(QMouseEvent *ev, bool click = false);
	void drawWaitMessage(QPainter &painter);

	// draw the background, inline function
	void fillBackground(QPainter &painter, const QRect& rect) {
		static QBrush brush(QColor(63, 31, 63), Qt::Dense4Pattern);
		painter.fillRect(rect, brush);
	}

	qreal scale;
	QTransform scaler, scalerI;
	QPixmap	pixmap;
};

#endif // SCALEDVIEW_H
