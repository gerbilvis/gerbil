#include "roidock.h"

#include <iostream>
#include "../gerbil_gui_debug.h"

static cv::Rect QRect2CVRect(const QRect &r) {
	return cv::Rect(r.x(), r.y(), r.width(), r.height());
}

static QRect CVRect2QRect(const cv::Rect &r) {
	return QRect(r.x, r.y, r.width, r.height);
}

RoiDock::RoiDock(QWidget *parent) :
	QDockWidget(parent)
{
	setupUi(this);
	initUi();
}

const QRect& RoiDock::getRoi() const
{
	return roiView->roi;
}

void RoiDock::setRoi(const cv::Rect &roi)
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

void RoiDock::setMaxBands(int bands)
{
	/* init bandsSlider according to maximum */
	bandsSlider->setMinimum(3);
	bandsSlider->setMaximum(bands);
	// default is no interpolation
	bandsSlider->setValue(bands);
}

void RoiDock::initUi()
{
	/* init signals */
	connect(roiButtons, SIGNAL(clicked(QAbstractButton*)),
			 this, SLOT(processRoiButtonsClicked(QAbstractButton*)));
	connect(roiView, SIGNAL(newSelection(const QRect&)),
			this, SLOT(processNewSelection(const QRect&)));
	connect(bandsSlider, SIGNAL(valueChanged(int)),
			this, SLOT(processBandsSliderChange(int)));
	connect(bandsSlider, SIGNAL(sliderMoved(int)),
			this, SLOT(processBandsSliderChange(int)));
}

void RoiDock::processBandsSliderChange(int b)
{
	bandsLabel->setText(QString("%1 bands").arg(b));
	if (!bandsSlider->isSliderDown()) {
		emit specRescaleRequested(b);
	}
}

void RoiDock::processRoiButtonsClicked(QAbstractButton *sender)
{
	QDialogButtonBox::ButtonRole role = roiButtons->buttonRole(sender);
	roiButtons->setDisabled(true);
	if (role == QDialogButtonBox::ResetRole) {
		resetRoi();
	} else if (role == QDialogButtonBox::ApplyRole) {
		applyRoi();
	}
}

void RoiDock::processNewSelection(const QRect &roi)
{
	curRoi = roi;
	roiButtons->setEnabled(true);

	QString title("<b>ROI:</b> %1, %2 - %3, %4 (%5x%6)");
	title = title.arg(roi.x()).arg(roi.y()).arg(roi.right()).arg(roi.bottom())
			.arg(roi.width()).arg(roi.height());
	roiTitle->setText(title);
}

void RoiDock::applyRoi()
{
	cv::Rect roi = QRect2CVRect(curRoi);
	emit roiRequested(roi);

	// reset will go back to current state
	oldRoi = curRoi;
}

void RoiDock::resetRoi()
{
	curRoi = oldRoi;
	roiView->roi = curRoi;
	processNewSelection(curRoi);
	roiView->update();
}


void RoiDock::updatePixmap(const QPixmap image)
{
	roiView->setPixmap(image);
	//GGDBGM(format("pixmap size %1%x%2%")%image.width() %image.height()<<endl);
	roiView->update();
}
