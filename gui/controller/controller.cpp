#include "controller.h"
#include "subscriptions.h"
#include "controller/distviewcontroller.h"
#include <imginput.h>
#include <rectangles.h>

#include "model/imagemodel.h"
#include "model/labelingmodel.h"
#include "model/illuminationmodel.h"
#include "model/falsecolormodel.h"
#include "model/graphsegmentationmodel.h"
#include "model/clusteringmodel.h"

#include "widgets/mainwindow.h"
#include "app/gerbilapplication.h" // to connect queue exception signal

#include <algorithm>

//#define GGDBG_MODULE
#include "gerbil_gui_debug.h"

Controller::Controller(const QString &filename,
                       bool limited_mode,
                       const QString &labelfile,
                       QObject *parent)

    : QObject(parent),
      // initialize all pointers so we don't access them too early w/o notice
      im(nullptr), lm(nullptr), fm(nullptr), illumm(nullptr),
      gsm(nullptr),
#ifdef WITH_SEG_MEANSHIFT
      cm(nullptr),
#endif
      dvc(nullptr),
      queuethread(nullptr)
{
	// reset internal ROI state tracking
	resetROISpawned();

	// start background task queue thread
	connect(&queue, SIGNAL(exception(std::exception_ptr, bool)),
	        GerbilApplication::instance(),
	        SLOT(handle_exception(std::exception_ptr,bool)),
	        Qt::BlockingQueuedConnection);
	startQueue();

	im = new ImageModel(queue, limited_mode, this);
	// load image
	cv::Rect dimensions = im->loadImage(filename);
	imgSize = cv::Size(dimensions.width, dimensions.height);

	// create gui (perform initUI before connecting signals!)
	window = new MainWindow();
	window->initUI(filename);

	// initialize models
	initImage();
	fm = new FalseColorModel(&queue);
	initFalseColor(); // depends on ImageModel / initImage()

	// The order of connection is crucial for fm and Controller.
	// fm needs to get the signal first. Otherwise it will
	// hand out invalid cached data.
	connect(im, SIGNAL(imageUpdate(representation::t,SharedMultiImgPtr,bool)),
	        fm, SLOT(processImageUpdate(representation::t,SharedMultiImgPtr,bool)));
	connect(im, SIGNAL(imageUpdate(representation::t,SharedMultiImgPtr,bool)),
	        this, SLOT(processImageUpdate(representation::t,SharedMultiImgPtr,bool)));

	lm = new LabelingModel(this);
	initLabeling(dimensions);
	illumm = new IllumModel(&queue, this);
	initIlluminant();
	gsm = new GraphSegmentationModel(&queue, this);
	initGraphSegmentation(); // depends on ImageModel / initImage()
#ifdef WITH_SEG_MEANSHIFT
	cm = new ClusteringModel(this);
#endif /* WITH_SEG_MEANSHIFT */

	// initialize sub-controllers (after initializing the models...)
	dvc = new DistViewController(this, &queue, im);
	dvc->init();

	// init dock widgets
	initDocks();

	// connect slots/signals
	window->initSignals(this, dvc);

	/* TODO: better place. But do not use init model functions:
	 * dvc are created after these are called
	 */
	connect(dvc, SIGNAL(requestOverlay(cv::Mat1b)),
	        this, SIGNAL(requestOverlay(cv::Mat1b)));
	connect(lm,
	        SIGNAL(newLabeling(const cv::Mat1s&, const QVector<QColor>&, bool)),
	        dvc, SLOT(updateLabels(cv::Mat1s,QVector<QColor>,bool)));
	connect(lm, SIGNAL(partialLabelUpdate(const cv::Mat1s&,const cv::Mat1b&)),
			dvc, SLOT(updateLabelsPartially(const cv::Mat1s&,const cv::Mat1b&)));
	connect(dvc, SIGNAL(alterLabelRequested(short,cv::Mat1b,bool)),
	        lm, SLOT(alterLabel(short,cv::Mat1b,bool)));
	connect(illumm, SIGNAL(newIlluminantCurve(QVector<multi_img::Value>)),
	        dvc, SIGNAL(newIlluminantCurve(QVector<multi_img::Value>)));
	connect(illumm, SIGNAL(newIlluminantApplied(QVector<multi_img::Value>)),
	        dvc, SIGNAL(newIlluminantApplied(QVector<multi_img::Value>)));

#ifdef WITH_SEG_MEANSHIFT
	connect(cm, SIGNAL(subscribeRepresentation(QObject*,representation::t)),
	        this, SLOT(subscribeRepresentation(QObject*,representation::t)));
	connect(cm, SIGNAL(unsubscribeRepresentation(QObject*,representation::t)),
	        this, SLOT(unsubscribeRepresentation(QObject*,representation::t)));
	connect(im, SIGNAL(imageUpdate(representation::t,SharedMultiImgPtr,bool)),
	        cm, SLOT(processImageUpdate(representation::t,SharedMultiImgPtr,bool)));
#endif

	connect(this, SIGNAL(preROISpawn(cv::Rect,cv::Rect,std::vector<cv::Rect>,std::vector<cv::Rect>,bool)),
	        dvc, SLOT(processPreROISpawn(cv::Rect,cv::Rect,std::vector<cv::Rect>,std::vector<cv::Rect>,bool)));
	connect(this, SIGNAL(postROISpawn(cv::Rect,cv::Rect,std::vector<cv::Rect>,std::vector<cv::Rect>,bool)),
	        dvc, SLOT(processPostROISpawn(cv::Rect,cv::Rect,std::vector<cv::Rect>,std::vector<cv::Rect>,bool)));

	/* start with initial label or provided labeling
	 * Do this after all signals are connected, and before initial ROI spawn!
	 */
	if (labelfile.isEmpty()) {
		lm->addLabel();
	} else {
		lm->loadLabeling(labelfile);
	}

	/* Initial ROI images spawning. Do it before showing the window but after
	 * all signals were connected! */
	//GGDBGM("dimensions " << dimensions << endl);
	roi = dimensions; // initial ROI is image size, except:
	if (roi.area() > 262144) {
		// image is bigger than 512x512, start with a smaller ROI in center
		roi.width = std::min(roi.width, 512);
		roi.height = std::min(roi.height, 512);
		roi.x = 0.5*(dimensions.width - roi.width);
		roi.y = 0.5*(dimensions.height - roi.height);
	}

	GGDBGM("roi " << roi  << endl);
	spawnROI();

	// The IMG representation must always be subscribed. Otherwise all the logic
	// in ImageModel fails. So we subscribe the Controller forever.
	subscribeRepresentation(this, representation::IMG);

	GGDBGM("init distview subscriptions" << endl);
	dvc->initSubscriptions();

	GGDBGM("init done, showing mainwindow" << endl);

	// we're done! show window
	window->show();
}

Controller::~Controller()
{
	// stop background task queue thread
	stopQueue();
	window->deleteLater();
}


/** Image management **/

// connect all signals between model and other parties
void Controller::initImage()
{
	// nothing
}

// depends on ImageModel
void Controller::initFalseColor()
{
	fm->setMultiImg(representation::IMG, im->getImage(representation::IMG));
	fm->setMultiImg(representation::GRAD, im->getImage(representation::GRAD));
}

void Controller::initIlluminant()
{
	illumm->setMultiImage(im->getFullImage());

	connect(illumm, SIGNAL(requestInvalidateROI(cv::Rect)),
	        this, SLOT(invalidateROI(cv::Rect)));
}

void Controller::initGraphSegmentation()
{
	gsm->setMultiImage(representation::IMG, im->getImage(representation::IMG));
	gsm->setMultiImage(representation::GRAD, im->getImage(representation::GRAD));

	connect(gsm, SIGNAL(alterLabelRequested(short,cv::Mat1b,bool)),
	        lm, SLOT(alterLabel(short,cv::Mat1b,bool)));
	// (gsm seedingDone <-> bandDock seedingDone connection in initDocks)
}

/** Labeling management **/

// connect all signals between model and other parties
void Controller::initLabeling(cv::Rect dimensions)
{
	// initialize label model
	lm->setImageSize(dimensions.height, dimensions.width);
}

void Controller::spawnROI(cv::Rect newRoi)
{
	const bool reuse = true;
	updateROI(reuse, newRoi);
}

void Controller::invalidateROI(cv::Rect newRoi)
{
	const bool reuse = false;
	updateROI(reuse, newRoi);
}

void Controller::rescaleSpectrum(int bands)
{
	// TODO: this was cancelTasks with current ROI. Now all tasks cancelled.
	// FIXME: This might break stuff!
	queue.cancelTasks();

	updateROI(false, cv::Rect(), bands);
}

void Controller::debugSubscriptions()
{
	//std::cerr << "** TYPE      subscribed flag" << std::endl;
	for (auto r : representation::all()) {
		std::cerr << "** " << std::left << std::setw(7) << r;
		if (subscriptions.images.subscribed(r)) {
			std::cerr << "    subscribed";
		} else {
			std::cerr <<  "not subscribed";
		}
		std::cerr << std::endl;
	}
}

void Controller::updateROI(bool reuse, cv::Rect newRoi, int bands)
{
	const cv::Rect oldRoi = roi;

	// no new ROI provided
	if (cv::Rect() == newRoi) {
		newRoi = roi ;
	} else {
		roi = newRoi;
	}
	resetROISpawned();
	GGDBGM("bands=" << bands << ", newRoi=" << newRoi << endl);

	// prepare incremental update and test worthiness
	std::vector<cv::Rect> sub, add;
	if (reuse) {
		// Compute if it is profitable to add/sub pixels given old and new ROI,
		// instead of full recomputation, and retrieve corresponding regions.
		bool profitable = rectTransform(im->getROI(), newRoi, sub, add);
		reuse = profitable;
	}

	/** FIRST STEP: recycle existing payload **/
	// Give other objects a chance to recycle old ROI image data.
	emit preROISpawn(oldRoi, roi, sub, add, reuse);

	if (!reuse) {
		// invalidate existing ROI information (to not re-use data)
		im->invalidateROI();
	}

	/** SECOND STEP: update metadata */
	lm->updateROI(newRoi);
	illumm->setRoi(newRoi);

	/** THIRD STEP: update payload */
	/* This has to be done in the right order!
	 * IMG before all others, GRAD before GRADPCA it is implicit here but we
	 * would like this knowledge to be part of image model's logic.
	 */
	for (auto r : representation::all()) {
		bool needed = subscriptions.images.subscribed(r);

		if (needed) {
			GGDBGM("     subscribed " << r << endl);
		} else {
			GGDBGM("not  subscribed " << r << endl);
		}

		if (needed) {
			/* tasks to (incrementally) re-calculate image data */
			im->spawn(r, newRoi, bands);
		}
	}

	emit postROISpawn(oldRoi, roi, sub, add, reuse);

	// FIXME: normTargetChanged() still necessary?
	// TODO: better method to make sure values in normalizationDock are correct
	// that means as soon as we have these values, report them directly to the
	// GUI.
	/*if (type == GRAD) {
		emit normTargetChanged(true);
	}*/
}

void Controller::subscribeImageBand(QObject *subscriber,
                                    representation::t repr,
                                    int bandId)
{
	// also subscribe to the relevant representation
	subscribeRepresentation(subscriber, repr);
	// only update if the subscription is new
	// TODO: what about other subscribers on bandId?
	bool newly = subscriptions.bands.subscribe(subscriber, BandId(repr, bandId));
	if (newly)
		im->computeBand(repr, bandId);
}

void Controller::unsubscribeImageBand(QObject *subscriber,
                                      representation::t repr,
                                      int bandId)
{
	subscriptions.bands.unsubscribe(subscriber, BandId(repr, bandId));
	unsubscribeRepresentation(subscriber, repr);
}

void Controller::subscribeFalseColor(QObject *subscriber,
                                     FalseColoring::Type coloring)
{
	//GGDBGM(coloring << endl);
	// also subscribe to the relevant representation
	subscribeRepresentation(subscriber, FalseColoring::basis(coloring));
	if (subscriptions.colorings.subscribe(subscriber, coloring)) {
		//GGDBGM("requesting from fm " << coloring << endl);
		fm->requestColoring(coloring);
	}
}

void Controller::unsubscribeFalseColor(QObject *subscriber,
                                       FalseColoring::Type coloring)
{
	//GGDBGM(coloring << endl);
	subscriptions.colorings.unsubscribe(subscriber, coloring);
	if (!subscriptions.colorings.subscribed(coloring)) {
		// no more subscriptions for coloring,
		// cancel computation if any.
		fm->cancelComputation(coloring);
	}
	unsubscribeRepresentation(subscriber, FalseColoring::basis(coloring));
}

void Controller::recalcFalseColor(FalseColoring::Type coloringType)
{
	if (subscriptions.colorings.subscribed((coloringType))) {
		fm->requestColoring(coloringType, /* recalc */ true);
	}
}

void Controller::subscribeRepresentation(QObject *subscriber,
                                         representation::t repr)
{
	if (subscriptions.images.subscribe(subscriber, repr)) {
		GGDBGM("new subscription, ");
		if (roiSpawned[repr]) {
			GGDBGP("RE-spawning ROI "<< roi << " for " << repr << endl);
			im->respawn(repr);
		} else {
			GGDBGP("   spawning ROI "<< roi << " for " << repr << endl);
			im->spawn(repr, roi, -1);
			roiSpawned[repr] = true;
		}
	}
}

void Controller::unsubscribeRepresentation(QObject *subscriber,
                                           representation::t repr)
{
	GGDBGM("unsubscribe " << repr << endl);
	subscriptions.images.unsubscribe(subscriber, repr);
}

void Controller::startQueue()
{
	// start worker thread
	//boost::ref h(queue);
	queuethread = new boost::thread(boost::ref(queue));
	// h is not needed anymore
}

void Controller::stopQueue()
{
	// cancel all jobs, then wait for thread to return
	queue.halt();
	queuethread->join();
	// queuethread is not a QObject, delete it here.
	delete queuethread;
	queuethread = 0;
}

// method for debugging focus
void Controller::focusChange(QWidget *old, QWidget *now)
{
	if (!old || !now)
		return;
	std::cerr << "Focus changed from " << old->objectName().toStdString()
	          << " to " << now->objectName().toStdString() << std::endl;
}

void Controller::processImageUpdate(representation::t repr,
                                    SharedMultiImgPtr image,
                                    bool duplicate)
{
	if (duplicate) {
		GGDBGM("duplicate update, ignoring" << endl);
		return;
	}

	roiSpawned[repr] = true;

	Subscription<BandId>::KeySet bandUpdates;

	for (auto s : subscriptions.bands) {
		if (repr == s.id.repr)	 {
			bandUpdates.insert(s.id);
		}
	}
	for (auto ib : bandUpdates) {
		//GGDBGM("requesting band " << ib.first << " " << ib.second << endl);
		im->computeBand(ib.repr, ib.band);
	}

	// false color updates

	typedef std::unordered_set<FalseColoring::Type, std::hash<int> >
	        FalseColoringSet;
	FalseColoringSet fcUpdates;
	for (auto s : subscriptions.colorings) {
		FalseColoring::Type coloring = s.id;
		if (FalseColoring::isBasedOn(coloring, repr)) {
			//GGDBGM("found subscriber for " << coloring <<
			//       " based on " << repr << endl);
			fcUpdates.insert(coloring);
		}
	}
	for (auto coloring : fcUpdates) {
		//GGDBGM("requesting from fm " << coloring << endl);
		emit pendingFalseColorUpdate(coloring);
		fm->requestColoring(coloring);
	}
}

void Controller::resetROISpawned()
{
	for (auto r : representation::all()) {
		roiSpawned[r] = false;
	}
}
