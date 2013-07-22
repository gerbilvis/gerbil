#ifndef AUTOHIDEWIDGET_H
#define AUTOHIDEWIDGET_H

#include <QWidget>
#include <QGraphicsView>
#include <QGraphicsProxyWidget>

// TODO: abstract version of ViewerController
class AutohideWidget : public QWidget
{
	Q_OBJECT
public:
	/* does not take a parent. Widgets in QGraphicsView must always be
	 * top-level (no parent).
	 */
	explicit AutohideWidget();

	/* this method is called with the corresponding proxy item as argument.
	 * the widget will use it as its anchor in the graphics scene.
	 */
	void initProxy(QGraphicsProxyWidget *proxy);
	
signals:
	
public slots:
	virtual void scrollIn();
	virtual void scrollOut();

protected:
	enum {
		STATE_IN = 1,
		STATE_OUT = 2
	} state;
	QGraphicsProxyWidget *proxy;

	virtual void timerEvent(QTimerEvent *e);
	virtual void changeEvent(QEvent *e);
};

/* our custom graphics view always resizes the scene accordingly */
class GraphicsView : public QGraphicsView
{
public:
	GraphicsView(QWidget *parent) : QGraphicsView(parent) {}

protected:
	void resizeEvent(QResizeEvent *event) {
		if (scene())
			scene()->setSceneRect(QRect(QPoint(0, 0), event->size()));
		QGraphicsView::resizeEvent(event);
	}
};

#endif // AUTOHIDEWIDGET_H
