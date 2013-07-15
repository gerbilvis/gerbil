#include "roidock.h"

#include <iostream>
#include "../gerbil_gui_debug.h"

static cv::Rect QRect2CVRect(const QRect &r) {
	return cv::Rect(r.x(), r.y(), r.width(), r.height());
}

static QRect CVRect2QRect(const cv::Rect &r) {
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
	if(roi == QRect2CVRect(curRoi)) {
		// GUI already up-to-date, prevent loop
		return;
	}

	curRoi = CVRect2QRect(roi);
	oldRoi = curRoi;
	roiView->roi = CVRect2QRect(roi);
	roiView->update();
	processNewSelection(CVRect2QRect(roi));
}

void ROIDock::initUi()
{
	connect(roiButtons, SIGNAL(clicked(QAbstractButton*)),
			 this, SLOT(processRoiButtonsClicked(QAbstractButton*)));
	connect(roiView, SIGNAL(newSelection(const QRect&)),
			this, SLOT(processNewSelection(const QRect&)));
}


void ROIDock::processRoiButtonsClicked(QAbstractButton *sender)
{
	QDialogButtonBox::ButtonRole role = roiButtons->buttonRole(sender);
	roiButtons->setDisabled(true);
	if (role == QDialogButtonBox::ResetRole) {
		resetRoi();
	} else if (role == QDialogButtonBox::ApplyRole) {
		applyRoi();
	}
}

void ROIDock::processNewSelection(const QRect &roi)
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
	processNewSelection(curRoi);
	roiView->update();
}


void ROIDock::updatePixmap(const QPixmap image)
{
	roiView->setPixmap(image);
	//GGDBGM(format("pixmap size %1%x%2%")%image.width() %image.height()<<endl);
	roiView->update();
}
