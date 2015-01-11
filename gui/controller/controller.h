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

#include <gerbil_cplusplus.h>

// forward declarations

class DistViewController;
class GraphSegmentationModel;
class FalseColorModel;
class LabelingModel;
class ClusteringModel;
class IllumModel;
class ImageModel;
class MainWindow;

class BandDock;
class NormDock;
class RoiDock;
class FalseColorDock;
class IllumDock;
class GraphSegWidget;
class ClusteringDock;
class LabelDock;

namespace seg_graphs
{
	class GraphSegConfig;
}

class Subscriptions;

/** Controller class. */
class Controller : public QObject
{
	Q_OBJECT
public:
	explicit Controller(
			const QString &filename, bool limited_mode,
			const QString &labelfile, QObject *parent = GBL_NULLPTR);
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

	/** This signal is emitted before the controller spawns a new ROI.
	 *
	 * The data supplied as arguments is useful for efficiently recycle image
	 * payload that is computed on a ROI basis.
	 * oldroi and newroi contain the old and new ROI areas as a cv::Rect
	 * respectively.
	 * sub and add vectors cointain the areas that need to be subtracted and
	 * added to create the new ROI from the old roi.
	 * reuse is true if the total area changed is smaller than the new ROI and
	 * the old ROI data may be recycled, i.e. it is true if it is less work to
	 * compute the differential update from the old to the new ROI and it is
	 * false when it is cheaper or mandatory to compute the new ROI data from
	 * scratch.
	 * If reuse is false, any data MUST be recomputed from scratch. This
	 * is due to operations changing data in the ROI, e.g. spectral rescaling
	 * or lighting.
	 */
	void preROISpawn(cv::Rect const & oldroi,
					 cv::Rect const & newroi,
					 std::vector<cv::Rect> const & sub,
					 std::vector<cv::Rect> const & add,
					 bool profitable
					 );

	/** This signal is emitted after he controller has spawned a new ROI. 
	 *
	 * The parameters are the same as with preROISpawn. Be aware that any
	 * image data in the image model for the old ROI may already have been
	 * invalidated. To access old ROI data the preROISpawn signal must be used.
	 */
	void postROISpawn(cv::Rect const & oldroi,
					  cv::Rect const & newroi,
					  std::vector<cv::Rect> const & sub,
					  std::vector<cv::Rect> const & add,
					  bool profitable
					  );


/// DOCKS

	// these are send to the graphSegModel
	void requestGraphseg(representation::t type,
						 cv::Mat1s seedMap,
						 const seg_graphs::GraphSegConfig &config,
						 bool resetLabel);
	void requestGraphsegBand(representation::t type, int bandId,
							 cv::Mat1s seedMap,
							 const seg_graphs::GraphSegConfig &config,
							 bool resetLabel);

	/** Let GUI elements know we have requested a FalseColor update
	 * for coloringType. */
	void pendingFalseColorUpdate(FalseColoring::Type coloringType);

public slots:
	// for debugging, activate by connecting it in main.cpp
	void focusChange(QWidget * old, QWidget * now);

	/** requests (from GUI) */
	// change ROI, effectively spawning new image data, reusing cached ROI data
	void spawnROI(cv::Rect newRoi = cv::Rect());
	// change ROI, spawn new image data, rebuild everything from scratch.
	void invalidateROI(cv::Rect newRoi = cv::Rect());
	// change number of bands, spawning new image data
	void rescaleSpectrum(int bands);

	void debugSubscriptions();

protected slots:
	// these are requested by the graphSegWidget
	void requestGraphseg(representation::t,
						 const seg_graphs::GraphSegConfig &config,
						 bool resetLabel);
	void requestGraphsegCurBand(const seg_graphs::GraphSegConfig &config,
								bool resetLabel);
	void highlightSingleLabel(short label, bool highlight);

	void processImageUpdate(representation::t repr,
							SharedMultiImgPtr image,
							bool duplicate);

/// SUBSCRIPTIONS

	// Subscriptions provide a way for GUI objects to tell the Controller
	// what kind of data they currently require. Based on the received
	// (un-)subscriptions the Controller computes the arising dependencies and
	// makes sure all necessary data is computed on the model end. See the
	// slots below for subscribable data.
	// This concept is to efficiently handle dependencies between the
	// different models and relieve GUI classes from this task.
	// GUI objects should immediately un-subscribe data if they do not need
	// it, i.e. when they are hidden or the user selects a different type of
	// instance of the data to be displayed. Doing so only a minumum of
	// required computations are done on the model end.
	// Frequent subscriptions and unsubscriptions triggering requests on the
	// model end are expected to be cheap. The models have to make sure valid
	// results are cached and thus frequent reqests for the same data are
	// cheap.

	/** Subscribe for image band bandId of representation repr.
	 *
	 * A subscriber connected to ImageModel::bandUpdate() will receive the
	 * signal for the given representation and band at least until it
	 * un-subscribes.  This does not imply that a non-subscribed object will
	 * not receive the signal. The subscription just guarantees the signal
	 * will be emitted. Any other object subscribed for the representation
	 * will trigger the emission of the signal as well.
	 */
	void subscribeImageBand(QObject *subscriber,
							representation::t repr,
							int bandId);
	/** Un-subscribe from image band from ImageModel.
	 *
	 * This signals to the controller that the subscriber does not depend on
	 * the ImageModel::bandUpdate() signal for the given band anymore. The
	 * signal may still be received due to other subscribed objects.
	 */
	void unsubscribeImageBand(QObject *subscriber,
							  representation::t repr,
							  int bandId);

	/** Subscribe for false color image updates from FalseColorModel.
	 *
	 * A subscriber connected to FalseColorModel::falseColoringUpdate() will
	 * receive the signal for the given false color type at least until it
	 * un-subscribes.  This does not imply that a non-subscribed object will
	 * not receive the signal. The subscription just guarantees the
	 * computation will be done and the signal will be emitted. Any other
	 * object subscribed for the representation will trigger the emission of
	 * the signal as well.
	 */
	void subscribeFalseColor(QObject *subscriber,
							 FalseColoring::Type coloring);
	/** Un-subscribe from false color representation from FalseColorModel.
	 *
	 * This signals to the controller that the subscriber does not depend on
	 * the FalseColorModel::falseColoringUpdate() signal for the given false
	 * color representation anymore. The signal may still be received due to
	 * other subscribed objects.
	 */
	void unsubscribeFalseColor(QObject *subscriber,
									  FalseColoring::Type coloring);
	/** Explicitly trigger a re-calculation of SOM based false color image,
	 * even if an  up-to-date cached instance exists (SOM is
	 * non-deterministic). */
	void recalcFalseColor(FalseColoring::Type coloringType);

	/** Subscribe for image representation from ImageModel.
	 *
	 * A subscriber connected to ImageModel::imageUpdate() will receive this
	 * signal for the subscribed representation at least until it
	 * un-subscribes. This does not imply that a non-subscribed object will
	 * not receive the signal. The subscription just guarantees the
	 * computation will be done signal will be emitted. Any other object
	 * subscribed for the representation will trigger the emission of the
	 * signal as well. 
	 */
	void subscribeRepresentation(QObject *subscriber,
								 representation::t repr);
	/** Un-subscribe from image representation from ImageModel.
	 *
	 * This signals to the controller that the subscriber does not depend on
	 * the ImageModel::imageUpdate() signal for the given representation
	 * anymore. The signal may still be received due to other subscribed
	 * objects.
	 */
	void unsubscribeRepresentation(QObject *subscriber,
								   representation::t repr);

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

	/** Update ROI geometry or update data in current ROI.
	 *
	 * This changes the geometry of the ROI if a new rect is given as
	 * argument. Additionaly, if the bands parameter is set, this follows a
	 * spectral rescale and the image data in the ROI is re-computed
	 * accordingly.
	 */
	void updateROI(bool reuse, cv::Rect newRoi = cv::Rect(), int bands = -1);

	/** Returns true if there is a subscriber for the given representation,
	 * otherwise returns false.
	 */
	bool haveSubscriber(representation::t type);

	/** Reset internal ROI state tracking: ROIs for all representation types
	 * are set to non-spawned. That is a ROI spawn for each subscribed
	 * representation is necessary for the GUI to become up-to-date.*/
	void resetROISpawned();

/// VIEWERS

	// main window (or gui slave)
	MainWindow *window;

/// MODELS

	// image model stores all multispectral image representations (IMG, GRAD,
	// ...)
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
	NormDock *normDock;
	RoiDock *roiDock;
	FalseColorDock *falseColorDock;
	IllumDock *illumDock;
	ClusteringDock *clusteringDock;
	LabelDock *labelDock;

/// DistViewController

	// setup distribution views and manage them and their models
	DistViewController *dvc;

/// QUEUE

	BackgroundTaskQueue queue;
	boost::thread *queuethread;

/// SUBSCRIPTIONS
	// The current ROI.
	cv::Rect roi;

	// see subscriptions.h
	Subscriptions *subs;

	// Track spawn state of ROIs for each representation.
	// true -> ROI for this representation has been spawned. This makes
	// it possible to decide between full ROI spawn or re-spawn. See
	// ImageModel::imageUpdate() signal.
	bool roiSpawned[representation::REPSIZE];

/// IMAGE INFO (READ ONLY)

	// Size of the loaded multi_img
	cv::Size imgSize;

};

#endif // CONTROLLER_H
