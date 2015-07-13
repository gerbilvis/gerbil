#ifndef MODEWIDGET_H
#define MODEWIDGET_H

#include "autohidewidget.h"
#include "ui_modewidget.h"

enum SelectionMode
{
	Zoom,
	Pick,
	Label
};


class AutohideView;
class QButtonGroup;

class ModeWidget : public AutohideWidget, private Ui::ModeWidget
{
	Q_OBJECT

public:
	explicit ModeWidget(AutohideView* view);
	~ModeWidget();

protected:
	void initUi();

signals:
	void modeChanged(SelectionMode m);

private slots:
	void on_zoomButton_released();
	void on_pickButton_released();
	void on_labelButton_released();

private:
	QButtonGroup *group;

};

#endif // MODEWIDGET_H
