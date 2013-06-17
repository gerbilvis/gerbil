#ifndef DOCKCONTROLLER_H
#define DOCKCONTROLLER_H

#include "../common/background_task.h"
#include "model/representation.h"
#include "../common/shared_data.h"

#include <QObject>
#include <QPixmap>

class ImageModel;
class IllumModel;
class FalseColorModel;

class Controller;

class MainWindow;

class ROIDock;
class RgbDock;
class IllumDock;
class GraphSegmentationDock;
class UsSegmentationDock;

#include <QObject>

class DockController : public QObject
{
	Q_OBJECT
public:
	explicit DockController(Controller *chief);
	

	void setImageModel(ImageModel *im) {this->im = im;}
	void setIllumModel(IllumModel *illumm) {this->illumm = illumm;}
	void setFalseColorModel(FalseColorModel *fm) {this->fm =fm;}
	void setMainWindow(MainWindow *mw) {this->mw = mw;}
	void init();
signals:
	void rgbRequested();
public slots:
	void enableDocks(bool enable, TaskType tt);
	void processRGB(QPixmap rgb);
protected:
	/* Create dock widget objects. */
	void createDocks();

	/* Initialize signal/slot connections. */
	void setupDocks();

	Controller* chief;
	
	ImageModel *im;
	IllumModel *illumm;
	FalseColorModel *fm;

	MainWindow *mw;

	ROIDock *roiDock;
	RgbDock *rgbDock;
	IllumDock *illumDock;
	GraphSegmentationDock *graphSegDock;
	UsSegmentationDock *usSegDock;
};

#endif // DOCKCONTROLLER_H

