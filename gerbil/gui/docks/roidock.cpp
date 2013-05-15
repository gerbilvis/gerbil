#include "roidock.h"

#include "../gerbil_gui_debug.h"

ROIDock::ROIDock(QWidget *parent) :
    DockWidget(parent)
{
	GGDBG_ENTER_LEAVE();
	ui.setupUi(this);
	initUi();
}

QRect ROIDock::getRoi() const
{
	GGDBG_ENTER_LEAVE();
	return ui.roiView->roi;
}

void ROIDock::setRoi(const QRect roi)
{
	GGDBG_ENTER_LEAVE();
	ui.roiView->roi = roi;
}

void ROIDock::initUi()
{
	GGDBG_ENTER_LEAVE();
	connect(ui.roiButtons, SIGNAL(clicked(QAbstractButton*)),
			 this, SLOT(roiButtonsClicked(QAbstractButton*)));
	connect(ui.roiView, SIGNAL(newSelection(QRect)),
			this, SLOT(newRoiSelected(QRect)));
}


void ROIDock::roiButtonsClicked(QAbstractButton *sender)
{
	GGDBG_ENTER_LEAVE();
	QDialogButtonBox::ButtonRole role = ui.roiButtons->buttonRole(sender);
	ui.roiButtons->setDisabled(true);
	if (role == QDialogButtonBox::ResetRole) {
		emit resetRoiClicked();
	} else if (role == QDialogButtonBox::ApplyRole) {
		emit applyRoiClicked();
	}

}

void ROIDock::newRoiSelected(const QRect roi)
{
	GGDBG_ENTER_LEAVE();
	ui.roiButtons->setEnabled(true);

	QString title("<b>ROI:</b> %1, %2 - %3, %4 (%5x%6)");
	title = title.arg(roi.x()).arg(roi.y()).arg(roi.right()).arg(roi.bottom())
			.arg(roi.width()).arg(roi.height());
	ui.roiTitle->setText(title);
}


void ROIDock::setPixmap(const QPixmap image)
{
	GGDBG_ENTER_LEAVE();
	ui.roiView->setPixmap(image);
}
