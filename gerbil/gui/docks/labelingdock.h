#ifndef LABELINGDOCK_H
#define LABELINGDOCK_H

#include <QDockWidget>

#include "ui_labelingdock.h"

class LabelingDock : public QDockWidget, protected Ui::LabelingDock
{
	Q_OBJECT
	
public:
	explicit LabelingDock(QWidget *parent = 0);
	~LabelingDock();

	void initUi();

signals:
	void requestLoadLabeling();
	void requestSaveLabeling();
};

#endif // LABELINGDOCK_H
