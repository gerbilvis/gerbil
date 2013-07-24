#ifndef RGBDOCK_H
#define RGBDOCK_H

#include <QDockWidget>
#include <shared_data.h>
#include "../model/representation.h"
#include "../model/falsecolor.h"
#include "ui_rgbdock.h"

class ScaledView;

class RgbDock : public QDockWidget, private Ui::RgbDock{
	Q_OBJECT
public:
	explicit RgbDock(QWidget *parent = 0);
	
signals:
	void rgbRequested(coloring type); // TODO: rename to falseColorRequested
	void rgbLazyRequested(coloring type); // TODO: rename to falseColorRequested

public slots:
	void processImageUpdate(representation::t type, SharedMultiImgPtr image);
	void processVisibilityChanged(bool visible);
	void updatePixmap(coloring type, QPixmap p); // TODO: add bool gradient
protected slots:
	void selectColorRepresentation();
	void calculateColorRepresentation();

protected:
	void initUi();
//	ScaledView *view;

	// type and gradient that is currently displayed (not status of the input fields)
	coloring displayType;
	bool displayGradient;

	// True if the current rgb image is up to date.
	bool rgbValid;

	// True if the dock is visible (that is tab is selected or top level).
	// Note: This is not the same as QWidget::isVisible()!
	bool dockVisible;
};

#endif // RGBDOCK_H
