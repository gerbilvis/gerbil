#include "controller.h"
#include <imginput.h>

#include <QFileInfo>
#include <cstdlib> // for exit()

Controller::Controller(QString filename, bool limited_mode)
	: im(limited_mode), queuethread(0)
{
	// background task queue thread
	startQueue();

	// load image
	cv::Rect dimensions = im.loadImage(filename);
	if (dimensions.x < 1) {
		exit(4); // Qt's exit does not work before calling exec();
	}
	if (dimensions.area() > 262144) {
		// image is bigger than 512x512, start with a smaller ROI
		dimensions.width = std::min(dimensions.width, 512);
		dimensions.height = std::min(dimensions.height, 512);
	}

	// create gui
	window = new MainWindow(limited_mode);

	// connect slots/signals
	window->initUI(this);
	initImage();
	initLabeling();

	/* Initial ROI images spawning. Do it before showing the window but after
	 * all signals were connected! */
	spawnROI(dimensions);

	// MODEL To be removed when refactored
	// into model classes.
	window->viewerContainer->image = &(this->image); // usw..
	viewerContainer->gradient = &gradient;
	viewerContainer->imagepca = &imagepca;
	viewerContainer->gradientpca = &gradientpca;
	viewerContainer->setTaskQueue(&queue);

	// set title and show window
	QFileInfo fi(filename.c_str());
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
			SIGNAL(bandSelected(representation, dim)),
			im, SLOT(computeBand(representation, dim)));
	connect(window, SIGNAL(rgbRequested()),
			im, SLOT(computeRGB()));

	/* im -> others */
	connect(im, SIGNAL(bandUpdate(QPixmap, QString)),
			window, SLOT(onBandUpdate(QPixmap, QString)));
	connect(im, SIGNAL(rgbUpdate(QPixmap)),
			window, SLOT(processRGB(QPixmap));
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

	im.spawn(roi, reuse);

	enableGUILater(true);
}

/** Labeling management **/

// connect all signals between model and other parties
void Controller::initLabeling()
{
	/* gui requests */
	connect(window, SIGNAL(clearLabelRequested(short)),
			lm, SLOT(alterLabel(short)));
	connect(window, SIGNAL(alterLabelRequested(short,cv::Mat1b,bool)),
			lm, SLOT(alterLabel(short,cv::Mat1b,bool)));

	/* lm -> others */
	connect(lm, SIGNAL(labelMatrix(cv::Mat1s)),
			window, SLOT(setLabelMatrix(cv::Mat1s)));
	connect(lm, SIGNAL(labelMatrix(cv::Mat1s)),
			window->getViewerContainer(), SLOT(setLabelMatrix(cv::Mat1s)));

	connect(lm, SIGNAL(newLabeling(const QVector<QColor> &, bool)),
			this, SLOT(propagateLabelingChange(const QVector<QColor> &, bool)));
	connect(lm, SIGNAL(newLabeling(const QVector<QColor> &, bool)),
			this, SLOT(processLabelingChange(const QVector<QColor> &, bool)));
}

void Controller::propagateLabelingChange(const QVector<QColor> &colors, bool changed)
{
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
