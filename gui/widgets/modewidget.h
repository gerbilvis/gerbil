#ifndef MODEWIDGET_H
#define MODEWIDGET_H

#include "bandview.h"
#include "autohidewidget.h"
#include "ui_modewidget.h"
#include "actionbutton.h"

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
	void inputModeChanged(ScaledView::InputMode m);
	void cursorSizeChanged(BandView::CursorSize s);

public slots:
	void updateInputMode(ScaledView::InputMode m);
	void updateCursorMode(BandView::CursorMode m);
	void updateCursorSize(BandView::CursorSize s);

	ActionButton* getRubberButton() { return rubberButton; }

private slots:
	void on_zoomButton_released();
	void on_pickButton_released();
	void on_labelButton_released();
	void on_smallCurButton_released();
	void on_mediumCurButton_released();
	void on_bigCurButton_released();
	void on_hugeCurButton_released();

private:
	QButtonGroup *modeGroup;
	QButtonGroup *cursorGroup;

	void setCursorButtonsVisible(bool visible);

};

#endif // MODEWIDGET_H
