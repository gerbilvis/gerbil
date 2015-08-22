#include "modewidget.h"

#include <QButtonGroup>

using IM = ScaledView::InputMode;
using CM = BandView::CursorMode;
using CS = BandView::CursorSize;

ModeWidget::ModeWidget(AutohideView *view) :
	AutohideWidget()
{
	setupUi(this);
}

ModeWidget::~ModeWidget()
{

}

void ModeWidget::initUi()
{
	/* ensure that only one button is selected at a time */
	modeGroup = new QButtonGroup();
	modeGroup->addButton(zoomButton);
	modeGroup->addButton(pickButton);
	modeGroup->addButton(labelButton);

	cursorGroup = new QButtonGroup();
	cursorGroup->addButton(smallCurButton);
	cursorGroup->addButton(mediumCurButton);
	cursorGroup->addButton(bigCurButton);
	cursorGroup->addButton(hugeCurButton);
	setCursorButtonsVisible(false);

	zoomButton->setAction(actionZoom);
	connect(actionZoom, SIGNAL(triggered()),
	        this, SLOT(zoomMode()));

	labelButton->setAction(actionLabel);
	connect(actionLabel, SIGNAL(triggered()),
	        this, SLOT(labelMode()));

	pickButton->setAction(actionPick);
	connect(actionZoom, SIGNAL(triggered()),
	        this, SLOT(pickMode()));

	rubberButton->setAction(actionRubber);
	overrideButton->setAction(actionOverride);
}

void ModeWidget::updateInputMode(ScaledView::InputMode m)
{
	setEnabled(true);
	switch (m) {
	case IM::Zoom:
		zoomButton->setChecked(true);
		setCursorButtonsVisible(false);
		break;
	case IM::Pick: pickButton->setChecked(true);
		setCursorButtonsVisible(false);
		break;
	case IM::Label:
		labelButton->setChecked(true);
		setCursorButtonsVisible(true);
		break;
	default:
		setEnabled(false);
		setCursorButtonsVisible(false);// a mode out of our reach
	}
}

void ModeWidget::updateCursorSize(BandView::CursorSize s)
{
	setEnabled(true);
	switch (s) {
	case CS::Small:
		smallCurButton->setChecked(true);
		break;
	case CS::Medium:
		mediumCurButton->setChecked(true);
		break;
	case CS::Big:
		bigCurButton->setChecked(true);
		break;
	case CS::Huge:
		hugeCurButton->setChecked(true);
		break;
	default:
		setEnabled(false);
	}
}

void ModeWidget::updateCursorMode(BandView::CursorMode m)
{
	rubberButton->setChecked(m == CM::Rubber);
}

void ModeWidget::zoomMode()
{
	setCursorButtonsVisible(false);
}

void ModeWidget::pickMode()
{
	setCursorButtonsVisible(false);
}

void ModeWidget::labelMode()
{
	setCursorButtonsVisible(true);
}

void ModeWidget::setCursorButtonsVisible(bool visible)
{
	labelModeWidget->setVisible(visible);
	adjustSize();
	reposition();
}

void ModeWidget::on_smallCurButton_released()
{
	emit cursorSizeChanged(CS::Small);
}

void ModeWidget::on_mediumCurButton_released()
{
	emit cursorSizeChanged(CS::Medium);
}

void ModeWidget::on_bigCurButton_released()
{
	emit cursorSizeChanged(CS::Big);
}

void ModeWidget::on_hugeCurButton_released()
{
	emit cursorSizeChanged(CS::Huge);
}
