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
	void mouseMoveEvent(QMouseEvent *ev) { cursorAction(ev); }
	void mousePressEvent(QMouseEvent *ev) { cursorAction(ev, true); }
	virtual void setPixmap(QPixmap p);
protected:
	virtual void cursorAction(QMouseEvent *ev, bool click = false);
	void drawWaitMessage(QPainter &painter);

	qreal scale;
	QTransform scaler, scalerI;
	QPixmap	pixmap;
};

#endif // SCALEDVIEW_H
