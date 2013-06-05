#ifndef RGBDOCK_H
#define RGBDOCK_H

#include <QDockWidget>

class ScaledView;

class RgbDock : public QDockWidget
{
	Q_OBJECT
public:
	explicit RgbDock(QWidget *parent = 0);
	
signals:
	
public slots:
	void updatePixmap(QPixmap p);
protected:
	void initUi();
	ScaledView *view;
};

#endif // RGBDOCK_H
