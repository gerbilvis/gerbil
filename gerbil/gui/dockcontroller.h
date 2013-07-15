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
class UsSegmentationModel;
class LabelingModel;

class Controller;

class MainWindow;

class BandDock;
class LabelingDock;
class NormDock;
class ROIDock;
class RgbDock;
class IllumDock;
class GraphSegWidget;
class UsSegmentationDock;

namespace vole
{
	class GraphSegConfig;
}

#include <QObject>

class DockController : public QObject
{
	Q_OBJECT
public:
	explicit DockController(Controller *chief, cv::Rect fullImgSize);
	void init();
signals:
	void rgbRequested();
	void requestGraphsegBand(representation::t type, int bandId,
							 const vole::GraphSegConfig &config,
							 bool resetLabel);
public slots:
	void enableDocks(bool enable, TaskType tt);
protected slots:
	void requestGraphsegCurBand(const vole::GraphSegConfig &config,
								bool resetLabel);
protected:
	/* Create dock widget objects. */
	void createDocks();

	/* Initialize signal/slot connections. */
	void setupDocks();

	Controller* chief;
	cv::Rect fullImgSize;

	BandDock *bandDock;
	LabelingDock *labelingDock;
	NormDock *normDock;
	ROIDock *roiDock;
	RgbDock *rgbDock;
	IllumDock *illumDock;
	UsSegmentationDock *usSegDock;
};

#endif // DOCKCONTROLLER_H

