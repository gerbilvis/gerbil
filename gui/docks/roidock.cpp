#include "roidock.h"
#include "../widgets/roiview.h"
#include "../widgets/autohidewidget.h"

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
	setObjectName("ROIDock");
	setupUi(this);
	initUi();
}

QRect RoiDock::getRoi() const
{
	return roiView->roi();
}

void RoiDock::setRoi(const cv::Rect &roi)
{
	if(roi == QRect2CVRect(curRoi)) {
		// GUI already up-to-date, prevent loop
		return;
	}

	// set curRoi, reflect it in GUI
	processNewSelection(CVRect2QRect(roi), true);

	// remember old selection for reset
	oldRoi = curRoi;
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
	// initialize ROI view
	view->init();
	roiView = new ROIView();
	view->setScene(roiView);
	connect(roiView, SIGNAL(newContentRect(QRect)),
			view, SLOT(fitContentRect(QRect)));

	// initialize button row
	btn = new AutohideWidget();
	uibtn = new Ui::RoiDockButtonUI();
	uibtn->setupUi(btn);
	roiView->offBottom = AutohideWidget::OutOffset;
	view->addWidget(AutohideWidget::BOTTOM, btn);

	/* init signals */
	connect(roiView, SIGNAL(newSizeHint(QSize)),
			view, SLOT(updateSizeHint(QSize)));

	connect(roiView, SIGNAL(newSelection(const QRect&)),
			this, SLOT(processNewSelection(const QRect&)));
	connect(bandsSlider, SIGNAL(valueChanged(int)),
			this, SLOT(processBandsSliderChange(int)));
	connect(bandsSlider, SIGNAL(sliderMoved(int)),
			this, SLOT(processBandsSliderChange(int)));

	applyAction = new QAction(this);
	applyAction->setText("Apply");
	uibtn->applyButton->setAction(applyAction);
	roiView->setApplyAction(applyAction);
	connect(applyAction, SIGNAL(triggered()), this, SLOT(applyRoi()));

	resetAction = new QAction(this);
	resetAction->setText("Reset");
	uibtn->resetButton->setAction(resetAction);
	roiView->setResetAction(resetAction);
	connect(resetAction, SIGNAL(triggered()), this, SLOT(resetRoi()));
}

void RoiDock::processBandsSliderChange(int b)
{
	bandsLabel->setText(QString("%1 bands").arg(b));
	if (!bandsSlider->isSliderDown()) {
		emit specRescaleRequested(b);
	}
}

void RoiDock::processNewSelection(const QRect &roi, bool internal)
{
	curRoi = roi;
	if (internal) {
		// also update the roiView
		roiView->setROI(roi);
	} else {
		// we have something to apply / reset from
		enableActions(true);
	}

	QString title("<b>ROI:</b> %1, %2 - %3, %4 (%5x%6)");
	title = title.arg(roi.x()).arg(roi.y()).arg(roi.right()).arg(roi.bottom())
			.arg(roi.width()).arg(roi.height());
	roiTitle->setText(title);
}

void RoiDock::applyRoi()
{
	enableActions(false);
	cv::Rect roi = QRect2CVRect(curRoi);
	emit roiRequested(roi);

	// reset will go back to current state
	oldRoi = curRoi;
}

void RoiDock::resetRoi()
{
	enableActions(false);
	curRoi = oldRoi;
	processNewSelection(curRoi, true);
}


void RoiDock::updatePixmap(const QPixmap image)
{
	roiView->setPixmap(image);
	//GGDBGM(format("pixmap size %1%x%2%")%image.width() %image.height()<<endl);
	roiView->update();
}

void RoiDock::enableActions(bool enable)
{
	applyAction->setEnabled(enable);
	resetAction->setEnabled(enable);
}
