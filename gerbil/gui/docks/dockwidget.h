#ifndef DOCKWIDGET_H
#define DOCKWIDGET_H

#include <QDockWidget>

class DockWidget : public QDockWidget
{
	Q_OBJECT
public:
	explicit DockWidget(QWidget *parent = 0);
	virtual ~DockWidget() {}
	
signals:
	
public slots:
	
};

#endif // DOCKWIDGET_H
