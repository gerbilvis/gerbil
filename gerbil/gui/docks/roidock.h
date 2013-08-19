#ifndef ROIDOCK_H
#define ROIDOCK_H

#include <ui_roidock.h>

#include <opencv2/core/core.hpp> // for cv::Rect
#include <QRect>
#include <QAbstractButton>

class RoiDock : public QDockWidget, private Ui::RoiDockUI
{
	Q_OBJECT
public:
	explicit RoiDock(QWidget *parent = 0);
	virtual ~RoiDock() {}
	
	const QRect &getRoi() const;


signals:
	/** The user has requested a new ROI by clicking apply. */
	void roiRequested(const cv::Rect &roi);

public slots:
	/** Update the pixmap displayed in the ROI-View. */
	void updatePixmap(const QPixmap image);
	/** Change the ROI in the GUI programmatically. */
	void setRoi(const cv::Rect &roi);

protected slots:
	void processRoiButtonsClicked(QAbstractButton *sender);
	// new roi selected in RoiView
	void processNewSelection(const QRect &roi);

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
};

#endif // ROIDOCK_H
