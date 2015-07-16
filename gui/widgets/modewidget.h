#ifndef MODEWIDGET_H
#define MODEWIDGET_H

#include "scaledview.h"
#include "autohidewidget.h"
#include "ui_modewidget.h"

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
	void modeChanged(ScaledView::InputMode m);

public slots:
	void updateMode(ScaledView::InputMode m);

private slots:
	void on_zoomButton_released();
	void on_pickButton_released();
	void on_labelButton_released();

private:
	QButtonGroup *group;

};

#endif // MODEWIDGET_H
