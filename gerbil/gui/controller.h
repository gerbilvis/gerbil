#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "model/image.h"
#include "model/labeling.h"
#include "model/illumination.h"
#include "model/falsecolor.h"
#include "model/ussegmentationmodel.h"
#include "mainwindow.h"
#include <background_task.h>
#include <background_task_queue.h>

#include <QObject>
#include <QMap>
#include <tbb/compat/thread>

class DockController;

class Controller : public QObject
{
	Q_OBJECT
public:
	explicit Controller(const std::string &filename, bool limited_mode);
	~Controller();
	
signals:

public slots:
	/** requests (from GUI) */
	// change ROI, effectively spawning new image data
	void spawnROI(const cv::Rect &roi);
	// change number of bands, spawning new image data
	void rescaleSpectrum(size_t bands);
	// change binnig to reflect, or not reflect, labeling
	void toggleLabels(bool toggle);

	// need additional label color
	void addLabel();
	// load a labeling from file
	void loadLabeling(const QString &filename = QString())
	{ lm.loadLabeling(filename); }
	// save a labeling to file
	void saveLabeling(const QString &filename = QString())
	{ lm.saveLabeling(filename); }

	/** notifies (from models, tasks) */
	// label colors added/changed
	void propagateLabelingChange(const cv::Mat1s &labels,
								 const QVector<QColor>& colors = QVector<QColor>(),
								 bool colorsChanged = false);

	// update image data in docks (to be moved to dockcontroller)
	void docksUpdateImage(representation::t type, SharedMultiImgPtr image);

	/** internal management (maybe make protected) */
	/* this function enqueues an empty task that will signal when all previous
	 * tasks are finished. the signal will trigger enableGUINow, and the
	 * GUI gets re-enabled at the right time.
	 * The roi argument is needed to specify if the other enqueued tasks are
	 * roi-bound. If somebody cancels all tasks with that roi, our task should
	 * be cancelled as-well, and re-enable not take place.
	 */
	void enableGUILater(bool withROI = false);
	// the other side of enableGUILater
	void enableGUINow(bool forreal);
	void disableGUI(TaskType tt = TT_NONE);

protected:
	// connect models with gui
	void initImage();
	void initIlluminant();
	void initLabeling();

	// init DockController
	void initDocks();

	// create background thread that processes BackgroundTaskQueue
	void startQueue();
	// stop and delete thread
	// (we did not test consecutive start/stop of the queue)
	void stopQueue();

	// update ROI, or its contents
	void updateROI(bool reuse,
				   cv::Rect roi = cv::Rect(), size_t bands = -1);

	// image model stores all multispectral image representations (IMG, GRAD, …)
	ImageModel im;

	// labeling model stores pixel/label associations and label color codes
	LabelingModel lm;

	// illumination model
	IllumModel illumm;

	/* false color model generates and stores RGB representations of
	 * multispectral data */
	FalseColorModel fm;

#ifdef WITH_SEG_MEANSHIFT
	// unsupervised segmentation model
	UsSegmentationModel um;
#endif /* WITH_SEG_MEANSHIFT */

	// setup dock widgets and manage interaction with models
	DockController *dc;

	// main window (or gui slave)
	MainWindow *window;

	BackgroundTaskQueue queue;
	std::thread *queuethread;

	/* A map of BackgroundTasks (QObject) to representations so that we know
	 * what representation a signaling task was working on */
	QMap<QObject*, representation> taskmap;
};

#endif // CONTROLLER_H
