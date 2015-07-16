#include "modewidget.h"

#include <QButtonGroup>

using IM = ScaledView::InputMode;

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

void ModeWidget::updateMode(ScaledView::InputMode m)
{
	setEnabled(true);
	switch (m) {
	case IM::Zoom: zoomButton->setChecked(true); break;
	case IM::Pick: pickButton->setChecked(true); break;
	case IM::Label: labelButton->setChecked(true); break;
	default:	setEnabled(false); // a mode out of our reach
	}
}

void ModeWidget::on_zoomButton_released()
{
	emit modeChanged(IM::Zoom);
}

void ModeWidget::on_pickButton_released()
{
	emit modeChanged(IM::Pick);
}

void ModeWidget::on_labelButton_released()
{
	emit modeChanged(IM::Label);
}
