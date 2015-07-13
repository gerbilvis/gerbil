#include "modewidget.h"

ModeWidget::ModeWidget(AutohideView *view) :
    AutohideWidget()
{
    setupUi(this);
}

ModeWidget::~ModeWidget()
{

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
