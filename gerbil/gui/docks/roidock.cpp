#include "roidock.h"

#include "../gerbil_gui_debug.h"

ROIDock::ROIDock(QWidget *parent) :
	DockWidget(parent)
{
	setupUi(this);
	initUi();
}

QRect ROIDock::getRoi() const
{
	return roiView->roi;
}

void ROIDock::setRoi(const QRect roi)
{
	oldRoi = roi;
	roiView->roi = roi;
	roiView->update();
}

void ROIDock::initUi()
{
	connect(roiButtons, SIGNAL(clicked(QAbstractButton*)),
			 this, SLOT(roiButtonsClicked(QAbstractButton*)));
	connect(roiView, SIGNAL(newSelection(QRect)),
			this, SLOT(newRoiSelected(QRect)));

}


void ROIDock::roiButtonsClicked(QAbstractButton *sender)
{
	QDialogButtonBox::ButtonRole role = roiButtons->buttonRole(sender);
	roiButtons->setDisabled(true);
	if (role == QDialogButtonBox::ResetRole) {
		emit resetRoiClicked();
		resetRoi();
	} else if (role == QDialogButtonBox::ApplyRole) {
		applyRoi();
		emit applyRoiClicked();
	}

}

void ROIDock::newRoiSelected(const QRect roi)
{
	curRoi = roi;
	roiButtons->setEnabled(true);

	QString title("<b>ROI:</b> %1, %2 - %3, %4 (%5x%6)");
	title = title.arg(roi.x()).arg(roi.y()).arg(roi.right()).arg(roi.bottom())
			.arg(roi.width()).arg(roi.height());
	roiTitle->setText(title);
}

void ROIDock::applyRoi()
{
	oldRoi = curRoi;
}

void ROIDock::resetRoi()
{
	curRoi = oldRoi;
	roiView->roi = curRoi;
	newRoiSelected(curRoi);
	roiView->update();
}


void ROIDock::setPixmap(const QPixmap image)
{
	roiView->setPixmap(image);
	roiView->update();
}
