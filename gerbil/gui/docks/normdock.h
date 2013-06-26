#ifndef NORMDOCK_H
#define NORMDOCK_H

#include <QDockWidget>

namespace Ui {
class NormDock;
}

class NormDock : public QDockWidget
{
	Q_OBJECT
	
public:
	explicit NormDock(QWidget *parent = 0);
	~NormDock();
	
private:
	Ui::NormDock *ui;
};

#endif // NORMDOCK_H
