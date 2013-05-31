#ifndef ROIDOCK_H
#define ROIDOCK_H

#include <ui_roidock.h>
#include "dockwidget.h"

#include <opencv2/core/core.hpp> // for cv::Rect
#include <QRect>
#include <QAbstractButton>

// FIXME: rename ROIDockUI -> ROIDock
class ROIDock : public DockWidget, private Ui::ROIDockUI
{
	Q_OBJECT
public:
	explicit ROIDock(QWidget *parent = 0);
	virtual ~ROIDock() {}
	
	const QRect &getRoi() const;
	void setRoi(const cv::Rect &roi); //TODO: slot?
	void setPixmap(const QPixmap image);

signals:
	void newSelection(const QRect &roi);
	void resetRoiClicked();
	void applyRoiClicked();
	// this one goes outside
	void roiRequested(const cv::Rect &roi);

public slots:
	void roiButtonsClicked(QAbstractButton *sender);
	void newRoiSelected(const QRect &roi);

protected:
	// helper functions to roiButtonsClicked
	void resetRoi();
	void applyRoi();

private:
	void initUi();

	QRect oldRoi;
	QRect curRoi;
};

#endif // ROIDOCK_H
