#ifndef DOCKCONTROLLER_H
#define DOCKCONTROLLER_H

#include "../common/background_task.h"
#include "model/representation.h"
#include "../common/shared_data.h"

#include <QObject>
#include <QPixmap>

class ImageModel;
class FalseColorModel;

class Controller;

class MainWindow;

class ROIDock;
class RgbDock;
class IllumDock;

#include <QObject>

class DockController : public QObject
{
	Q_OBJECT
public:
	explicit DockController(Controller *chief);
	

	void setImageModel(ImageModel *im) {this->im = im;}
	void setFalseColorModel(FalseColorModel *fm) {this->fm =fm;}
	void setMainWindow(MainWindow *mw) {this->mw = mw;}
	void init();
signals:
	void rgbRequested();
public slots:
	void enableDocks(bool enable, TaskType tt);
	void processNewImageData(representation::t type, SharedMultiImgPtr image);
	void processRGB(QPixmap rgb);
	void setRgbVisible(bool visible);
protected:
	/* Create dock widget objects. */
	void createDocks();

	/* Initialize signal/slot connections. */
	void setupDocks();

	Controller* chief;
	
	ImageModel *im;
	FalseColorModel *fm;

	MainWindow *mw;

	ROIDock *roiDock;
	RgbDock *rgbDock;
	IllumDock *illumDock;

	// FalseColor currently does not do any reference counting of users.
	// Therefore managing state of the rgb image is handled here in the
	// controller.
	bool rgbVisible;
	bool rgbImageValid;
};

#endif // DOCKCONTROLLER_H

