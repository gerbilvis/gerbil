#ifndef SCALEDVIEW_H
#define SCALEDVIEW_H

#include <QGLWidget>

class ScaledView : public QGLWidget
{
	Q_OBJECT
public:
	ScaledView(QWidget *parent = 0);
	void resizeEvent(QResizeEvent * ev);
	virtual void paintEvent(QPaintEvent *ev);
	void mouseMoveEvent(QMouseEvent *ev) { cursorAction(ev); }
	void mousePressEvent(QMouseEvent *ev) { cursorAction(ev, true); }
	virtual void setPixmap(const QPixmap &pixmap);

protected:
	virtual void cursorAction(QMouseEvent *ev, bool click = false);

	qreal scale;
	QTransform scaler, scalerI;
	const QPixmap *pixmap;
};

#endif // SCALEDVIEW_H
