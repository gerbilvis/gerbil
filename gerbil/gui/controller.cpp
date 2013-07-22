#include "controller.h"
#include "dockcontroller.h"
#include "dist_view/distviewcontroller.h"
#include <imginput.h>

#include "gerbil_gui_debug.h"

#include <cstdlib> // for exit()

// for DEBUG
std::ostream &operator<<(std::ostream& os, const cv::Rect& r)
{
	os << boost::format("%1%x%2%+%3%+%4%") % r.x % r.y % r.width % r.height;
	return os;
}

Controller::Controller(const std::string &filename, bool limited_mode,
	const QString &labelfile)
	: im(queue, limited_mode), fm(this, &queue), illumm(this),
	  gsm(this, &queue), queuethread(0),
	  dc(0), dvc(0) // so we don't access them too early
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
	window->initUI(filename, im.getNumBandsFull());

	// initialize models
	initImage();
	initFalseColor(); // depends on ImageModel / initImage()
	initIlluminant();
	initGraphSegmentation(); // depends on ImageModel / initImage()
	initLabeling(dimensions);

#ifdef WITH_SEG_MEANSHIFT
	um.setMultiImage(im.getImage(representation::IMG));
#endif /* WITH_SEG_MEANSHIFT */

	// initialize sub-controllers (after initializing the models...)
	dc = new DockController(this);
	dc->init();
	dvc = new DistViewController(this, &queue);
	dvc->init();

	// connect slots/signals
	window->initSignals(this, dvc);

	/* TODO: better place. Do not use init model functions, dvc is created later
	 */
	connect(dvc, SIGNAL(bandSelected(representation::t, int)),
			&im, SLOT(computeBand(representation::t, int)));
	connect(dvc, SIGNAL(requestOverlay(cv::Mat1b)),
			this, SIGNAL(requestOverlay(cv::Mat1b)));
	connect(&lm,
			SIGNAL(newLabeling(const cv::Mat1s&, const QVector<QColor>&, bool)),
			dvc, SLOT(updateLabels(cv::Mat1s,QVector<QColor>,bool)));
	connect(&lm, SIGNAL(partialLabelUpdate(const cv::Mat1s&,const cv::Mat1b&)),
			dvc, SLOT(updateLabelsPartially(const cv::Mat1s&,const cv::Mat1b&)));
	connect(dvc, SIGNAL(alterLabelRequested(short,cv::Mat1b,bool)),
			&lm, SLOT(alterLabel(short,cv::Mat1b,bool)));
	connect(&illumm, SIGNAL(newIlluminant(cv::Mat1f)),
			dvc, SIGNAL(newIlluminant(cv::Mat1f)));
	connect(&illumm, SIGNAL(illuminantIsApplied(bool)),
			dvc, SIGNAL(toggleIlluminantApplied(bool)));

	/* start with initial label or provided labeling
	 * Do this after all signals are connected, and before initial ROI spawn!
	 */
	if (labelfile.isEmpty()) {
		lm.addLabel();
	} else {
		lm.loadLabeling(labelfile);
	}

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

	// we're done! show window
	window->show();
}

Controller::~Controller()
{
	delete window;

	delete dc;
	delete dvc;

	// background task queue thread
	stopQueue();
}


/** Image management **/

// connect all signals between model and other parties
void Controller::initImage()
{
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

	connect(&illumm, SIGNAL(requestInvalidateROI(cv::Rect)),
			this, SLOT(invalidateROI(cv::Rect)));

	/* TODO: models should not request this! the controller of the model has
	 * to guard its operations!
	 */
	connect(&illumm, SIGNAL(setGUIEnabledRequested(bool,TaskType)),
			this, SLOT(setGUIEnabled(bool, TaskType)), Qt::DirectConnection);
}

void Controller::initGraphSegmentation()
{
	gsm.setMultiImage(representation::IMG, im.getImage(representation::IMG));
	gsm.setMultiImage(representation::GRAD, im.getImage(representation::GRAD));

	connect(&gsm, SIGNAL(alterLabelRequested(short,cv::Mat1b,bool)),
			&lm, SLOT(alterLabel(short,cv::Mat1b,bool)));

	connect(&gsm, SIGNAL(setGUIEnabledRequested(bool,TaskType)),
			this, SLOT(setGUIEnabled(bool, TaskType)));

	// (gsm seedingDone <-> bandDock seedingDone connection in initDocks)
}

/** Labeling management **/

// connect all signals between model and other parties
void Controller::initLabeling(cv::Rect dimensions)
{
	// initialize label model
	lm.setDimensions(dimensions.height, dimensions.width);
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

void Controller::rescaleSpectrum(int bands)
{
	queue.cancelTasks(im.getROI());
	disableGUI(TT_BAND_COUNT);

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

	updateROI(reuse, roi);

	enableGUILater(true);
}

void Controller::updateROI(bool reuse, cv::Rect roi, int bands)
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
			sets[i] = dvc->subImage(i, sub, roi);
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
			dvc->addImage(i, sets[i], add, roi);
		} else {
			dvc->setImage(i, im.getImage(i), roi);
		}
	}

	// TODO: better method to make sure values in normalizationDock are correct
	// that means as soon as we have these values, report them directly to the
	// GUI.
	/*if (type == GRAD) {
		emit normTargetChanged(true);
	}*/
}

void Controller::setGUIEnabled(bool enable, TaskType tt)
{
	/** for enable, this just re-enables everything
	 * for disable, this typically disables everything except the sender, so
	 * that the user can re-decide on that aspect or sth.
	 * it is a bit strange
	 */
	window->setGUIEnabled(enable, tt);

	// tell other controllers
	dc->setGUIEnabled(enable, tt);
	dvc->setGUIEnabled(enable, tt);
}

/** Tasks and queue thread management */

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
