#include "controller.h"
#include <imginput.h>

#include <QFileInfo>
#include <cstdlib> // for exit()

Controller::Controller(const std::string &filename, bool limited_mode)
	: im(queue, limited_mode), queuethread(0)
{
	// load image
	cv::Rect dimensions = im.loadImage(filename);
	if (dimensions.width < 1) {
		exit(4); // Qt's exit does not work before calling exec();
	}

	// background task queue thread
	startQueue();

	// initialize label model
	lm.setDimensions(dimensions.width, dimensions.height);

	// create gui (perform initUI before connecting signals)
	window = new MainWindow(limited_mode);
	window->initUI(dimensions, im.getSize());

	// connect slots/signals
	window->initSignals(this);
	initImage();
	initLabeling();

	// MODEL To be removed when refactored
	// into model classes.
	window->getViewerContainer()->setTaskQueue(&queue);

	// start with initial label (do this after signals, and before spawnROI()!
	lm.addLabel();

	/* Initial ROI images spawning. Do it before showing the window but after
	 * all signals were connected! */
	cv::Rect roi = dimensions; // initial ROI is image size, except:
	if (roi.area() > 262144) {
		// image is bigger than 512x512, start with a smaller ROI
		roi.width = std::min(roi.width, 512);
		roi.height = std::min(roi.height, 512);
	}
	spawnROI(roi);

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
			SIGNAL(bandSelected(representation, int)),
			&im, SLOT(computeBand(representation, int)));
	connect(window, SIGNAL(rgbRequested()),
			&im, SLOT(computeRGB()));

	/* im -> others */
	connect(&im, SIGNAL(bandUpdate(QPixmap, QString)),
			window, SLOT(changeBand(QPixmap, QString)));
	connect(&im, SIGNAL(rgbUpdate(QPixmap)),
			window, SLOT(processRGB(QPixmap)));
}

void Controller::spawnROI(const cv::Rect &roi)
{
	bool reuse = true;
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

	disableGUI(TT_SELECT_ROI);
	// TODO: check
	window->getViewerContainer()->disconnectAllViewers();

	updateROI(reuse, roi);

	enableGUILater(true);
}

void Controller::rescaleSpectrum(size_t bands)
{
	queue.cancelTasks(im.getROI());
	disableGUI(TT_BAND_COUNT);
	// TODO: check
	window->getViewerContainer()->disconnectAllViewers();

	updateROI(false, cv::Rect(), bands);

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
	QMap<representation, sets_ptr> sets;
	if (reuse) {
		foreach (representation i, im.representations) {
			sets[i] = window->getViewerContainer()->subImage(i, sub, roi);
		}
	}

	/** SECOND STEP: update metadata */

// TODO
//	updateRGB(true);
//	rgbDock->setEnabled(true);
	lm.updateROI(roi);

	/** THIRD STEP: update payload */
	/* this has to be done in the right order!
	 * IMG before all others, GRAD before GRADPCA
	 * it is implicit here but we would like this knowledge to be part of
	 * image model's logic
	 */
	foreach (representation i, im.representations) {

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

/** Labeling management **/

// connect all signals between model and other parties
void Controller::initLabeling()
{
	/* gui requests */
	connect(window, SIGNAL(clearLabelRequested(short)),
			&lm, SLOT(alterLabel(short)));
	connect(window, SIGNAL(alterLabelRequested(short,cv::Mat1b,bool)),
			&lm, SLOT(alterLabel(short,cv::Mat1b,bool)));

	/* lm -> others */
	connect(&lm, SIGNAL(labelingMatrix(cv::Mat1s)),
			window, SLOT(setLabelMatrix(cv::Mat1s)));
	connect(&lm, SIGNAL(labelingMatrix(cv::Mat1s)),
			window->getViewerContainer(), SLOT(setLabelMatrix(cv::Mat1s)));

	connect(&lm, SIGNAL(newLabeling(const QVector<QColor> &, bool)),
			this, SLOT(propagateLabelingChange(const QVector<QColor> &, bool)));
	connect(&lm, SIGNAL(partialLabelUpdate(cv::Mat1b, cv::Mat1s)),
			window->getViewerContainer(),
			SLOT(updateLabelsPartially(cv::Mat1b,cv::Mat1s)));

}

void Controller::propagateLabelingChange(const QVector<QColor> &colors, bool changed)
{
	window->processLabelingChange(colors, changed);

	if (changed)
		disableGUI();

	// TODO: we will talk to ViewerController directly
	window->getViewerContainer()->updateLabelColors(colors, changed);

	if (changed)
		enableGUILater();
}

void Controller::addLabel()
{
	int index = lm.addLabel();

	// select our new label for convenience
	window->selectLabel(index);
}

/** Tasks and queue thread management */
void Controller::toggleLabels(bool toggle)
{
	// TODO: is this really legit? I doubt.
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
	if (forreal)
		window->setGUIEnabled(true);
}

void Controller::disableGUI(TaskType tt)
{
	window->setGUIEnabled(false, tt);
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
