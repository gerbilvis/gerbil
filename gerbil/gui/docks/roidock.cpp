#include "roidock.h"

#include "../gerbil_gui_debug.h"

cv::Rect QRect2CVRect(const QRect &r) {
	return cv::Rect(r.x(), r.y(), r.width(), r.height());
}
QRect CVRect2QRect(const cv::Rect &r) {
	return QRect(r.x, r.y, r.width, r.height);
}

ROIDock::ROIDock(QWidget *parent) :
	DockWidget(parent)
{
	setupUi(this);
	initUi();
}

const QRect& ROIDock::getRoi() const
{
	return roiView->roi;
}

void ROIDock::setRoi(const cv::Rect &roi)
{
	// reset will go back to current state
	oldRoi = CVRect2QRect(roi);

	// start off our selector on new oldRoi
	roiView->roi = oldRoi;
	roiView->update();
}

void ROIDock::initUi()
{
	connect(roiButtons, SIGNAL(clicked(QAbstractButton*)),
			 this, SLOT(roiButtonsClicked(QAbstractButton*)));
	connect(roiView, SIGNAL(newSelection(const QRect&)),
			this, SLOT(newRoiSelected(const QRect&)));
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
	}
}

void ROIDock::newRoiSelected(const QRect &roi)
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
	cv::Rect roi = QRect2CVRect(curRoi);
	emit roiRequested(roi);

	// reset will go back to current state
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
