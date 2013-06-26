#ifndef BANDDOCK_H
#define BANDDOCK_H

#include <QDockWidget>

namespace Ui {
class BandDock;
}

class BandDock : public QDockWidget
{
	Q_OBJECT
	
public:
	explicit BandDock(QWidget *parent = 0);
	~BandDock();
	
private:
	Ui::BandDock *ui;
};

#endif // BANDDOCK_H
