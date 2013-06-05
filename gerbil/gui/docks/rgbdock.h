#ifndef RGBDOCK_H
#define RGBDOCK_H

#include <QDockWidget>
#include "../../common/shared_data.h"
#include "../model/representation.h"

class ScaledView;

class RgbDock : public QDockWidget
{
	Q_OBJECT
public:
	explicit RgbDock(QWidget *parent = 0);
	
signals:
	void rgbRequested();
public slots:
	void processImageUpdate(representation::t type, SharedMultiImgPtr image);
	void processVisibilityChanged(bool visible);
	void updatePixmap(QPixmap p);
protected:
	void initUi();
	ScaledView *view;

	// True if the current rgb image is up to date.
	bool rgbValid;
};

#endif // RGBDOCK_H
