#include "controller.h"
#include "dockcontroller.h"
#include <imginput.h>

#include "gerbil_gui_debug.h"

#include <QFileInfo>
#include <cstdlib> // for exit()

// for DEBUG
std::ostream &operator<<(std::ostream& os, const cv::Rect& r)
{
	os << boost::format("%1%x%2%+%3%+%4%") % r.x % r.y % r.width % r.height;
	return os;
}

Controller::Controller(const std::string &filename, bool limited_mode)
	: im(queue, limited_mode), fm(this, &queue), illumm(this), gsm(this, &queue),
	   queuethread(0), spectralRescaleInProgress(false)
{
	// load image
	cv::Rect dimensions = im.loadImage(filename);
	if (dimensions.width < 1) {
		exit(4); // Qt's exit does not work before calling exec();
	}

	// background task queue thread
	startQueue();

	// create gui (perform initUI before connecting signals!)
	window = new MainWindow(limited_mode);
	window->initUI(dimensions, im.getSize());

	// connect slots/signals
	window->initSignals(this);
	connect(window, SIGNAL(setGUIEnabledRequested(bool,TaskType)),
			this, SLOT(setGUIEnabled(bool, TaskType)));

	// initialize models
	initImage();
	initFalseColor(); // depends on ImageModel / initImage()
	initIlluminant();
	initGraphSegmentation(); // depends on ImageModel / initImage()
	initLabeling(dimensions);

#ifdef WITH_SEG_MEANSHIFT
	um.setMultiImage(im.getImage(representation::IMG));
#endif /* WITH_SEG_MEANSHIFT */

	// initialize docks (after initializing the models...)
	dc = new DockController(this);
	dc->init();

	// MODEL To be removed when refactored
	// into model classes.
	window->getViewerContainer()->setTaskQueue(&queue);

	// start with initial label (do this after signals, and before spawnROI())!
	lm.addLabel();

	/* Initial ROI images spawning. Do it before showing the window but after
	 * all signals were connected! */
	//GGDBGM("dimensions " << dimensions << endl);
	cv::Rect roi = dimensions; // initial ROI is image size, except:
	if (roi.area() > 262144) {
		// image is bigger than 512x512, start with a smaller ROI
		roi.width = std::min(roi.width, 512);
		roi.height = std::min(roi.height, 512);
	}
	//GGDBGM("roi " << roi << endl);
	spawnROI(roi);

	// obsolete
//	// compute data ranges. Currently only for normDock.
//	im.computeDataRange(representation::IMG);
//	im.computeDataRange(representation::GRAD);

	// set title and show window
	QFileInfo fi(QString::fromStdString(filename));
	window->setWindowTitle(QString("Gerbil - %1").arg(fi.completeBaseName()));
	window->show();
}

Controller::~Controller()
{
	delete window;
	// background task queue thread
	stopQueue();
}


/** Image management **/

// connect all signals between model and other parties
void Controller::initImage()
{
	/* gui requests */
	connect(window->getViewerContainer(),
			SIGNAL(bandSelected(representation::t, int)),
			&im, SLOT(computeBand(representation::t, int)));
}

// depends on ImageModel
void Controller::initFalseColor()
{
	fm.setMultiImg(representation::IMG, im.getImage(representation::IMG));

	connect(&im, SIGNAL(imageUpdate(representation::t,SharedMultiImgPtr)),
			&fm, SLOT(processImageUpdate(representation::t,SharedMultiImgPtr)));
}

void Controller::initIlluminant()
{
	illumm.setTaskQueue(&queue);
	illumm.setMultiImage(im.getFullImage());

	// signals illumm <-> viewer container
	connect(&illumm, SIGNAL(newIlluminant(cv::Mat1f)),
			window->getViewerContainer(), SLOT(newIlluminant(cv::Mat1f)));
	connect(&illumm, SIGNAL(illuminantIsApplied(bool)),
			window->getViewerContainer(), SLOT(setIlluminantApplied(bool)));

	connect(&illumm, SIGNAL(requestInvalidateROI(cv::Rect)),
			this, SLOT(invalidateROI(cv::Rect)));

	connect(&illumm, SIGNAL(setGUIEnabledRequested(bool,TaskType)),
			this, SLOT(setGUIEnabled(bool, TaskType)), Qt::DirectConnection);
}

void Controller::initGraphSegmentation()
{
	gsm.setMultiImage(representation::IMG, im.getImage(representation::IMG));
	gsm.setMultiImage(representation::GRAD, im.getImage(representation::GRAD));

	connect(&gsm, SIGNAL(alterLabelRequested(short,cv::Mat1b,bool)),
			&lm, SLOT(alterLabel(short,cv::Mat1b,bool)));

	connect(&im, SIGNAL(bandUpdate(QPixmap,QString)),
			&gsm, SLOT(setCurBand(QPixmap,QString)));

	connect(&gsm, SIGNAL(setGUIEnabledRequested(bool,TaskType)),
			this, SLOT(setGUIEnabled(bool, TaskType)));

	// gsm seedingDone <-> bandDock seedingDone connection in initDocks
}

/** Labeling management **/

// connect all signals between model and other parties
void Controller::initLabeling(cv::Rect dimensions)
{
	// initialize label model
	lm.setDimensions(dimensions.height, dimensions.width);

	/* gui requests */
	connect(window, SIGNAL(clearLabelRequested(short)),
			&lm, SLOT(alterLabel(short)));
	connect(window, SIGNAL(alterLabelRequested(short,cv::Mat1b,bool)),
			&lm, SLOT(alterLabel(short,cv::Mat1b,bool)));

	/* lm -> others */
	connect(&lm,
			SIGNAL(newLabeling(const cv::Mat1s&, const QVector<QColor>&, bool)),
			this, SLOT(propagateLabelingChange(
					 const cv::Mat1s&, const QVector<QColor> &, bool)));
	connect(&lm, SIGNAL(partialLabelUpdate(const cv::Mat1s&,const cv::Mat1b&)),
			window->getViewerContainer(),
			SLOT(updateLabelsPartially(const cv::Mat1s&,const cv::Mat1b&)));
}

void Controller::spawnROI(cv::Rect roi)
{
	const bool reuse = true;
	doSpawnROI(reuse, roi);
}

void Controller::invalidateROI(cv::Rect roi)
{
	const bool reuse = false;
	doSpawnROI(reuse, roi);
}

void Controller::rescaleSpectrum(size_t bands)
{
	queue.cancelTasks(im.getROI());
	disableGUI(TT_BAND_COUNT);
	// TODO: check
    // 2013-06-19 altmann: seems to work without disconnect as well.
	window->getViewerContainer()->disconnectAllViewers();

	spectralRescaleInProgress = true;

	updateROI(false, cv::Rect(), bands);

	enableGUILater(true);
}

void Controller::doSpawnROI(bool reuse, const cv::Rect &roi)
{
	// TODO: make a method cancelAllComputation that does following two steps
	/* TODO: this results in huge computation burden even if none of the
	 * running tasks would interfere. We could think of a more fine-grained
	 * method here. Previous was to keep track of "running task type"
	 *
	 * TODO: talk to petr about why we need the ROI state it seems to be only
	 * in use for this cancel and nothing else?
	 */
	if (!queue.isIdle()) {
		queue.cancelTasks(im.getROI());
		/* as we cancelled any tasks, we expect the image data not to reflect
		 * desired configuration, so we will recompute from scratch */
		reuse = false;
	}
	// also cancel CommandRunners
	fm.reset();

	disableGUI(TT_SELECT_ROI);
	// TODO: check
	// 2013-06-19 altmann: seems to work without disconnect as well.
	window->getViewerContainer()->disconnectAllViewers();

	updateROI(reuse, roi);

	enableGUILater(true);
}

void Controller::updateROI(bool reuse, cv::Rect roi, size_t bands)
{
	// no new ROI provided
	if (roi == cv::Rect())
		roi = im.getROI();

	// prepare incremental update and test worthiness
	std::vector<cv::Rect> sub, add;
	if (reuse) {
		/* compute if it is profitable to add/sub pixels given old and new ROI,
		 * instead of full recomputation, and retrieve corresponding regions
		 */
		bool profitable = MultiImg::Auxiliary::rectTransform(im.getROI(), roi,
															 sub, add);
		if (!profitable)
			reuse = false;
	} else {
		// invalidate existing ROI information (to not re-use data)
		im.invalidateROI();
	}

	/** FIRST STEP: recycle existing payload **/
	QMap<representation::t, sets_ptr> sets;
	if (reuse) {
		foreach (representation::t i, representation::all()) {
			sets[i] = window->getViewerContainer()->subImage(i, sub, roi);
		}
	}

	/** SECOND STEP: update metadata */

	lm.updateROI(roi);
	illumm.setRoi(roi);

	/** THIRD STEP: update payload */
	/* this has to be done in the right order!
	 * IMG before all others, GRAD before GRADPCA
	 * it is implicit here but we would like this knowledge to be part of
	 * image model's logic
	 */
	foreach (representation::t i, representation::all()) {

		/* tasks to (incrementally) re-calculate image data */
		im.spawn(i, roi, bands);

		/* tasks to (incrementally) update distribution view */
		if (reuse) {
			window->getViewerContainer()->addImage(i, sets[i], add, roi);
		} else {
			window->getViewerContainer()->setImage(i, im.getImage(i), roi);
		}
	}

	// TODO: better method to make sure values in normalizationDock are correct
	// that means as soon as we have these values, report them directly to the
	// GUI.
	/*if (type == GRAD) {
		emit normTargetChanged(true);
	}*/
}

void Controller::propagateLabelingChange(const cv::Mat1s& labels,
										 const QVector<QColor> &colors,
										 bool colorsChanged)
{
	// -> now a slot in bandDock (lm -> banddock)
	//window->processLabelingChange(labels, colors, colorsChanged);

	bool grandUpdate = !labels.empty() || colorsChanged;

	if (grandUpdate)
		disableGUI();

	// TODO: we will talk to ViewerController directly
	window->getViewerContainer()->updateLabels(labels, colors, colorsChanged);

	if (grandUpdate)
		enableGUILater();
}

void Controller::setGUIEnabled(bool enable, TaskType tt)
{
	/** for enable, this just re-enables everything
	 * for disable, this typically disables everything except the sender, so
	 * that the user can re-decide on that aspect or sth.
	 * it is a bit strange
	 */
	window->setGUIEnabled(enable, tt);

	// tell dock controller
	emit requestEnableDocks(enable, tt);
}
/** Tasks and queue thread management */
void Controller::toggleLabels(bool toggle)
{
	// TODO: is this really legit? I doubt.
	// This is only to apply changes instantly,
	// instead of waiting for queue.
	queue.cancelTasks();
	disableGUI(TT_TOGGLE_LABELS);

	window->getViewerContainer()->toggleLabels(toggle);

	enableGUILater(false);
}

void Controller::enableGUILater(bool withROI)
{
	BackgroundTask *t = (withROI ? new BackgroundTask(im.getROI())
								 : new BackgroundTask());
	BackgroundTaskPtr taskEpilog(t);
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)),
		this, SLOT(enableGUINow(bool)), Qt::QueuedConnection);
	queue.push(taskEpilog);
}

void Controller::enableGUINow(bool forreal)
{
	if(spectralRescaleInProgress) {
		// The number of spectral bands changed - let the GUI know about it.
		int nbands = im.getNumBandsROI();
		//GGDBGM(format("emitting nSpectralBandsChanged(%1%)")%nbands << endl);
		emit nSpectralBandsChanged(nbands);
		spectralRescaleInProgress = false;
	}
	if (forreal)
		setGUIEnabled(true);
}

void Controller::disableGUI(TaskType tt)
{
	setGUIEnabled(false, tt);
}

/* container to allow passing an object reference to std::thread()
 * needed by initQueue(), without this std::thread() would run on a copy
 */
template<typename T>
struct holder {
	holder(T& payload) : payload(payload) {}
	void operator()() { payload(); }

	T& payload;
};

void Controller::startQueue()
{
	// start worker thread
	holder<BackgroundTaskQueue> h(queue);
	queuethread = new std::thread(h);
	// h is not needed anymore
}

void Controller::stopQueue()
{
	// cancel all jobs, then wait for thread to return
	queue.halt();
	queuethread->join();
	delete queuethread;
	queuethread = 0;
}
