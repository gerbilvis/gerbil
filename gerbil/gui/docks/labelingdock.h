#ifndef LABELINGDOCK_H
#define LABELINGDOCK_H

#include <QDockWidget>

namespace Ui {
class LabelingDock;
}

class LabelingDock : public QDockWidget
{
	Q_OBJECT
	
public:
	explicit LabelingDock(QWidget *parent = 0);
	~LabelingDock();
	
private:
	Ui::LabelingDock *ui;
};

#endif // LABELINGDOCK_H
