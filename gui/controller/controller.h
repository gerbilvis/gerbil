#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <background_task/background_task.h>
#include <background_task/background_task_queue.h>

#include <QObject>
#include <QMap>
#include <boost/thread.hpp>

class DockController;
class DistViewController;
class GraphSegmentationModel;
class FalseColorModel;
class LabelingModel;
class ClusteringModel;
class IllumModel;
class ImageModel;
class MainWindow;

class Controller : public QObject
{
	Q_OBJECT
public:
	explicit Controller(const std::string &filename, bool limited_mode,
						const QString &labelfile);
	~Controller();

	MainWindow* mainWindow() { return window; }
	ImageModel* imageModel() { return im; }
	LabelingModel* labelingModel() {return lm; }
	FalseColorModel* falseColorModel() { return fm; }
	IllumModel* illumModel() { return illumm; }
	GraphSegmentationModel* graphSegmentationModel() { return gsm; }
	ClusteringModel* clusteringModel() { return cm; }

signals:

	void currentLabelChanged(int);

	/* pass-through to our other controller friends */
	void toggleIgnoreLabels(bool);
	void toggleSingleLabel(bool);
	void singleLabelSelected(int);

	void showIlluminationCurve(bool);

	void requestPixelOverlay(int y, int x);

	/* pass-through from our other controller friends */
	void requestOverlay(const cv::Mat1b&);

public slots:
	// for debugging, activate by connecting it in main.cpp
	void focusChange(QWidget * old, QWidget * now);

	/** requests (from GUI) */
	// change ROI, effectively spawning new image data, reusing cached ROI data
	void spawnROI(cv::Rect roi = cv::Rect());
	// change ROI, spawn new image data, rebuild everything from scratch.
	void invalidateROI(cv::Rect roi = cv::Rect());
	// change number of bands, spawning new image data
	void rescaleSpectrum(int bands);

	/** Enable and disable GUI as indicated by enable flag.
	 *
	 * When disable is requested all GUI elements other than those involved
	 * with the task indicated by TaskType whill be disabled. This is to enable
	 * the user to trigger a new operation of the same TaskType before the
	 * previous one is finished.  In this case the model in charge should
	 * cancel the ongoing calculations and re-start with the new user input.
	 */
	void setGUIEnabled(bool enable, TaskType tt = TT_NONE);

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
	void initFalseColor();
	void initIlluminant();
	void initGraphSegmentation();
	void initLabeling(cv::Rect dimensions);

	// create background thread that processes BackgroundTaskQueue
	void startQueue();
	// stop and delete thread
	// (we did not test consecutive start/stop of the queue)
	void stopQueue();

	void doSpawnROI(bool reuse, const cv::Rect &roi);

	// update ROI, or its contents
	void updateROI(bool reuse, cv::Rect roi = cv::Rect(), int bands = -1);

/// VIEWERS

	// main window (or gui slave)
	MainWindow *window;

/// MODELS

	// image model stores all multispectral image representations (IMG, GRAD, …)
	ImageModel *im;

	/* labeling model stores pixel/label associations and label color codes */
	LabelingModel *lm;

	/* false color model generates and stores RGB representations of
	 * multispectral data */
	FalseColorModel *fm;

	/* illumnation model performs illumination changes */
	IllumModel *illumm;

	/* graph segmentation model performs supervised segmentation */
	GraphSegmentationModel *gsm;

#ifdef WITH_SEG_MEANSHIFT
	/* clustering model provides global clustering */
	ClusteringModel *cm;
#endif /* WITH_SEG_MEANSHIFT */

/// OTHER CONTROLLERS

	// setup dock widgets and manage interaction with models
	DockController *dc;

	// setup distribution views and manage them and their models
	DistViewController *dvc;


/// QUEUE

	BackgroundTaskQueue queue;
	boost::thread *queuethread;
};

#endif // CONTROLLER_H
