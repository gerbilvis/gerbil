#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <model/representation.h>
#include <shared_data.h>
#include <background_task/background_task.h>
#include <background_task/background_task_queue.h>
#include <model/falsecolor/falsecoloring.h>

#include <QObject>
#include <QMap>
#include <boost/thread.hpp>


// forward declarations

class DockController;
class DistViewController;
class GraphSegmentationModel;
class FalseColorModel;
class LabelingModel;
class ClusteringModel;
class IllumModel;
class ImageModel;
class MainWindow;

class BandDock;
class LabelingDock;
class NormDock;
class RoiDock;
class FalseColorDock;
class IllumDock;
class GraphSegWidget;
class ClusteringDock;
class LabelDock;

namespace vole
{
	class GraphSegConfig;
}

class Subscriptions;

/** Controller class. */
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

/// DOCKS

	// these are send to the graphSegModel
	void requestGraphseg(representation::t type,
						 cv::Mat1s seedMap,
						 const vole::GraphSegConfig &config,
						 bool resetLabel);
	void requestGraphsegBand(representation::t type, int bandId,
							 cv::Mat1s seedMap,
							 const vole::GraphSegConfig &config,
							 bool resetLabel);

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
	void setGUIEnabledDocks(bool enable, TaskType tt = TT_NONE);

	/** internal management (maybe make protected) */
	/* this function enqueues an empty task that will signal when all previous
	 * tasks are finished. the signal will trigger enableGUINow, and the
	 * GUI gets re-enabled at the right time.
	 * The roi argument is needed to specify if the other enqueued tasks are
	 * roi-bound. If somebody cancels all tasks with that roi, our task should
	 * be cancelled as-well, and re-enable not take place.
	 */
	void enableGUILater();
	// the other side of enableGUILater
	void enableGUINow(bool forreal);
	void disableGUI(TaskType tt = TT_NONE);

protected slots:
	// these are requested by the graphSegWidget
	void requestGraphseg(representation::t,
						 const vole::GraphSegConfig &config,
						 bool resetLabel);
	void requestGraphsegCurBand(const vole::GraphSegConfig &config,
								bool resetLabel);
	void highlightSingleLabel(short label, bool highlight);

	void processImageUpdate(representation::t repr);

/// SUBSCRIPTIONS

	// Subscriptions help manage data dependencies between GUI elements. GUI
	// elements subscribe for computation results they need to display. The
	// controller triggers a re-calculation of results if the input data,
	// usually the selected ROI, changes for subscribed results.
	// Subscription is usually triggered by GUI-element visibility and user
	// selection of results. GUI elements are _not_ responsible to chache
	// result data but the model is. Subscribing to existing results is
	// expected to be fast. Usually this is achieved by caching results on the
	// model end.
	// This structure simplifies GUI and model logic since the controller
	// takes care of who is subscribed to what and neither the models nor the
	// GUI widgets have to track ROI changes and data dependencies.

	/** Subscribe for image band bandId of representation repr.
	 *
	 * If a GUI element (subscriber) is subscribed for an image band, i.e. the
	 * grayscale image of one spectral band, the controller requests a
	 * computation for the band from the image model whenever necessary, e.g.
	 * when the image data changes. 
	 * A subscriber receives updates via the ImageModel::bandUpdate() signal.
	 */
	void processSubscribeImageBand(QObject *subscriber, representation::t repr, int bandId);
	void processUnsubscribeImageBand(QObject *subscriber, representation::t repr, int bandId);

	/** Subscribes a subscriber for false color image updates from FalseColorModel. */
	void processSubscribeFalseColor(QObject *subscriber, FalseColoring::Type coloring);
	void processUnsubscribeFalseColor(QObject *subscriber, FalseColoring::Type coloring);
	/** Re-calculate a SOM based false color image even if an  up-to-date
	 * cached instance exists (SOM is non-deterministic). */
	void processRecalcFalseColor(FalseColoring::Type coloringType);

protected:
	// connect models with gui
	void initImage();
	void initFalseColor();
	void initIlluminant();
	void initGraphSegmentation();
	void initLabeling(cv::Rect dimensions);

	/** Create and setup docks. */
	void initDocks();
	void createDocks();
	/** Setup signal/slot connections. */
	void setupDocks();

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

/// DOCKS

	BandDock *bandDock;
	LabelingDock *labelingDock;
	NormDock *normDock;
	RoiDock *roiDock;
	FalseColorDock *falseColorDock;
	IllumDock *illumDock;
	ClusteringDock *clusteringDock;
	LabelDock *labelDock;

/// OTHER CONTROLLERS

	// setup dock widgets and manage interaction with models
	DockController *dc;

	// setup distribution views and manage them and their models
	DistViewController *dvc;

/// QUEUE

	BackgroundTaskQueue queue;
	boost::thread *queuethread;

/// SUBSCRIPTIONS

	// see subscriptions.h
	Subscriptions *subs;

};

#endif // CONTROLLER_H
