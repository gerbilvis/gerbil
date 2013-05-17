#ifndef VIEWPORTCONTROL_H
#define VIEWPORTCONTROL_H

#include "ui_viewportcontrol.h"
#include "compute.h"
#include <QMenu>
#include <QTimer>

class Viewport;
class multi_img_viewer;

class ViewportControl : public QWidget, private Ui::ViewportControl
{
	Q_OBJECT

public:
	explicit ViewportControl(multi_img_viewer *parent = 0);
	void init(Viewport *vp);

	int getBinCount() { return binSlider->value(); }
	void setType(representation type);

	QVector<QColor> labelColors;

public slots:
	void showLimiterMenu();
	void updateLabelColors(QVector<QColor> colors);
	void setAlpha(int alpha);
	void setBinCount(int n);
	void scrollIn();
	void scrollOut();

protected:
	enum {
		STATE_IN = 1,
		STATE_OUT = 2
	} state;

	void timerEvent(QTimerEvent *e);
	void changeEvent(QEvent *e);
	multi_img_viewer *holder;
	Viewport *viewport;

	QMenu limiterMenu;
};

#endif // VIEWPORTCONTROL_H
