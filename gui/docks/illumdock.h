#ifndef ILLUMDOCK_H
#define ILLUMDOCK_H

#include <QDockWidget>
#include "ui_illumdock.h"


class IllumDock : public QDockWidget, private Ui::IllumDock
{
	Q_OBJECT
	
public:
	explicit IllumDock(QWidget *parent = 0);
	~IllumDock();
protected slots:
	// from GUI
	void onIllum1Selected(int idx);
	void onIllum2Selected(int idx);
	void onShowToggled(bool show);
	void onApplyClicked();

signals:
	void applyIllum(); // TODO: rename
	void illum1Selected(int idx);
	void illum2Selected(int idx);
	/* effect: if(show): illuminat curve shown in viewers .*/
	void showIlluminationCurveChanged(bool show);
private:
	void initUi();
};

#endif // ILLUMDOCK_H
