#ifndef ROIDOCK_H
#define ROIDOCK_H

#include "ui_roidock.h"
#include "ui_roidock_buttons.h"

#include <opencv2/core/core.hpp> // for cv::Rect
#include <QRect>
#include <QAbstractButton>

class ROIView;
class AutohideWidget;

class RoiDock : public QDockWidget, private Ui::RoiDockUI
{
	Q_OBJECT
public:
	explicit RoiDock(QWidget *parent = 0);
	virtual ~RoiDock() {}
	
	QRect getRoi() const;


signals:
	/** The user has requested a new ROI by clicking apply. */
	void roiRequested(const cv::Rect &roi);

	/** User has requested a new binning by adjusting bands slider */
	void specRescaleRequested(int bands);

public slots:
	/** Update the pixmap displayed in the ROI-View. */
	void updatePixmap(const QPixmap image);
	/** Change the ROI in the GUI programmatically. */
	void setRoi(const cv::Rect &roi);
	/** Set maximum number of bands for binning slider */
	void setMaxBands(int bands);

protected slots:
	// band slider movement
	void processBandsSliderChange(int b);

	void processRoiButtonsClicked(QAbstractButton *sender);

	// new roi selected in RoiView or propagated through controller (internal)
	void processNewSelection(const QRect &roi, bool internal = false);

protected:
	// helper functions to roiButtonsClicked
	void resetRoi();
	void applyRoi();

private:
	void initUi();

	// The old ROI before apply (used for reset).
	QRect oldRoi;
	// The current ROI selected, but possibly not yet applied.
	QRect curRoi;

	// our viewport (a scene actually)
	ROIView *roiView;
	// UI and widget for our button row
	Ui::RoiDockButtonUI *uibtn;
	AutohideWidget *btn;
};

#endif // ROIDOCK_H
