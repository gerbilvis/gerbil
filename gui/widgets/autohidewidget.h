#ifndef AUTOHIDEWIDGET_H
#define AUTOHIDEWIDGET_H

#include <QWidget>
#include <QGraphicsProxyWidget>

class AutohideWidget : public QWidget
{
	Q_OBJECT
public:
	enum border {
		LEFT, RIGHT, TOP, BOTTOM
	};

	/* does not take a parent. Widgets in QGraphicsView must always be
	 * top-level (no parent).
	 */
	explicit AutohideWidget();

	/* this method is called with the corresponding proxy item as argument.
	 * the widget will use it as its anchor in the graphics scene.
	 * @loc the border this widget resides in
	 */
	void init(QGraphicsProxyWidget *proxy, border loc);
	
	/* stretch the widget to fill the whole border */
	void adjustToSize(QSize size);

	/* decide if scrolling needed based on mouse position */
	void triggerScrolling(QPoint pos);

signals:
	
public slots:
	virtual void scrollIn();
	virtual void scrollOut();

protected:
	// which border we are in
	border location;

	// visibility state
	enum {
		STATE_IN = 1, // scrolling into/residing in view
		STATE_OUT = 2 // scrolling out of view / hidden
	} state;

	// our connection to the graphics view
	QGraphicsProxyWidget *proxy;

	virtual void timerEvent(QTimerEvent *e);
	virtual void changeEvent(QEvent *e);
};

#endif // AUTOHIDEWIDGET_H
