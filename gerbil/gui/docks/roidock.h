#ifndef ROIDOCK_H
#define ROIDOCK_H

#include <QAbstractButton>

#include "dockwidget.h"
#include <ui_roidock.h>

class ROIDock : public DockWidget, private Ui::ROIDockUI
{
	Q_OBJECT
public:
	explicit ROIDock(QWidget *parent = 0);
	virtual ~ROIDock() {}
	
	QRect getRoi() const;
	void setRoi(const QRect roi);
	void setPixmap(const QPixmap image);
signals:
	void newSelection(QRect roi);
	void resetRoiClicked();
	void applyRoiClicked();
public slots:
	void roiButtonsClicked(QAbstractButton *sender);
	void newRoiSelected(const QRect roi);
private:
	void initUi();
};

#endif // ROIDOCK_H
