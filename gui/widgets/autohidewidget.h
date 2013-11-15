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

	// visibility state
	enum scrollstate {
		SCROLL_IN, // scrolling into/residing in view
		SCROLL_OUT, // scrolling out of view / hidden
		STAY_IN, // scroll into view and stay there
		STAY_OUT // scroll out and keep out
	};

	/* does not take a parent. Widgets in QGraphicsView must always be
	 * top-level (no parent).
	 */
	explicit AutohideWidget();

	/* current widget state */
	scrollstate scrollState() { return state; }

	/* this method is called with the corresponding proxy item as argument.
	 * the widget will use it as its anchor in the graphics scene.
	 * @loc the border this widget resides in
	 */
	void init(QGraphicsProxyWidget *proxy, border loc);
	
	/* adjust position and stretch the widget to fill the whole border */
	void reposition();

	/* decide if scrolling needed based on mouse position */
	void triggerScrolling(QPoint pos);

	/* Call this instead of QWidget::adjustSize()
	 * Whenever the widget gets a new size, we also have to reset its position.
	 */
	void adjust();

	static const int OutOffset = 10;

signals:
	
public slots:
	// get into view, stay there indefinitely if enforce is set
	virtual void scrollIn(bool enforce = false);
	// get out of view, stay there indefinitely if enforce is set
	virtual void scrollOut(bool enforce = false);

protected:
	// which border we are in
	border location;
	scrollstate state;

	// our connection to the graphics view
	QGraphicsProxyWidget *proxy;

	virtual void timerEvent(QTimerEvent *e);
	virtual void changeEvent(QEvent *e);
};

#endif // AUTOHIDEWIDGET_H
