#ifndef RGBDOCK_H
#define RGBDOCK_H

#include <QDockWidget>
#include "../../common/shared_data.h"
#include "../model/representation.h"
#include "../model/falsecolor.h"

class ScaledView;

class RgbDock : public QDockWidget
{
	Q_OBJECT
public:
	explicit RgbDock(QWidget *parent = 0);
	
signals:
	void rgbRequested(coloring type);
public slots:
	void processImageUpdate(representation::t type, SharedMultiImgPtr image);
	void processVisibilityChanged(bool visible);
	void updatePixmap(coloring type, QPixmap p);
protected:
	void initUi();
	ScaledView *view;

	// True if the current rgb image is up to date.
	bool rgbValid;

	// True if the dock is visible (that is tab is selected or top level).
	// Note: This is not the same as QWidget::isVisible()!
	bool dockVisible;
};

#endif // RGBDOCK_H
