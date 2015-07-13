#include "modewidget.h"

#include <QButtonGroup>

ModeWidget::ModeWidget(AutohideView *view) :
	AutohideWidget()
{
	setupUi(this);
	initUi();
}

ModeWidget::~ModeWidget()
{

}

void ModeWidget::initUi()
{
	/* ensure that only one button is selected at a time */
	group = new QButtonGroup();
	group->addButton(zoomButton);
	group->addButton(pickButton);
	group->addButton(labelButton);
}

void ModeWidget::on_zoomButton_released()
{
	SelectionMode m = Zoom;
	emit modeChanged(m);
}

void ModeWidget::on_pickButton_released()
{
	SelectionMode m = Pick;
	emit modeChanged(m);
}

void ModeWidget::on_labelButton_released()
{
	SelectionMode m = Label;
	emit modeChanged(m);
}
