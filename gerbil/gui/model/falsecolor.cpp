#include "falsecolor.h"

#include <background_task_queue.h>
#include <multi_img.h>
#include <multi_img_tasks.h>
#include <qtopencv.h>
#include <rgb.h>
#include <shared_data.h>
#include <tasks/rgbtbb.h>

#include <QImage>
#include <opencv2/core/core.hpp>

// all representation parameters are currently ignored or expected to be IMG

// RGB:
//  progressUpdate() in SOM einbauen,
//      SOM *som = SOMTrainer::train(config.som, img);

//  \-> wie kompiliert der header ohne .h datei? eig per class ProgressObs;...

// if a CommandRunner currently fails, this type of image can not be calculated
// until reset() is called


// Long term TODOs:
// Langfristige Frage: Sind QImages oder QPixmaps interessant? (oder beides),
//      was davon soll nur 1x im model sein?
//  \-> "QPixmaps cannot be directly shared between threads"
//       \-> model sollte aber im *einen* GUI-Thread sein.

// "init rgb.configs, if non default setup is neccessary"

// sichergehen, dass img immer der aktuelle ROI ausschnitt ist
// ROI per signal slot verteilen <-> code georg --> sollte dann hier egal sein,
//      weil das img ja immer passend gesetzt werden sollte

//  \-> Im output speichern, welcher algorithmus berechnet wurde, ist nicht
//      gerade toll...
//       \-> fuers RGB modul wuerde es besser passen, wenn model auch mit
//           gerbil::rgbalg arbeiten wuerde...
//      Alternative: QSignalMapper, passt gut zum enum, der kann aber nur
//           Signale ohne Parameter
//       \-> Wenn das Ergebnis im Parameter uebergeben wird geht's nicht
//       \-> output map im CommandRunner speichern? -> parameterloses signal

FalseColorModel::FalseColorModel(BackgroundTaskQueue *queue)
	: queue(queue)
{
	for (int i = 0; i < COLSIZE; ++i) {
#ifndef WITH_EDGE_DETECT
		if (i == SOM)
			continue;
#endif
		payload *p = new payload(representation::IMG, (coloring)i);
		map.insert((coloring)i, p);

		// FIXME:
		// QObject::connect: Cannot queue arguments of type 'FalseColorModel::coloring'
		// (Make sure 'FalseColorModel::coloring' is registered using qRegisterMetaType().)
		QObject::connect(
			p,    SIGNAL(calculationComplete(FalseColorModel::coloring, QImage)),
			this, SIGNAL(calculationComplete(FalseColorModel::coloring, QImage)),
			Qt::QueuedConnection);
	}
}

void FalseColorModel::setMultiImg(representation::t type,
								  SharedMultiImgPtr shared_img)
{
	// in the future, we might be interested in the other ones as well.
	if (type == representation::IMG) {
		this->shared_img = shared_img;
		reset();
	}
}

FalseColorModel::~FalseColorModel()
{
	cancel();
}

void FalseColorModel::reset()
{
	// in case we have some computation going on
	cancel();

	// reset all images
	PayloadList l = map.values();
	foreach(payload *p, l) {
		p->img = QImage();
		p->calcImg = qimage_ptr(new SharedData<QImage>(new QImage()));
		p->calcInProgress = false;
		/* TODO: maybe send the empty image as signal to
		 * disable viewing of obsolete information
		 * maybe add bool flag, so this doesn't happen at construction
		 */
	}
}

// if Tasks / CommandRunners are running, just cancel them, do not report back
// to widgets if the image is changed, after setMultiImg is called, all widgets
// will request a new img
void FalseColorModel::cancel()
{
	// terminate command runners
	emit terminateRunners();

	// tasks in queue are expected to be cancelled by a controller at this point

	// wait & destroy all runners
	foreach (payload *p, map) {
		if (p->runner != NULL) {
			p->runner->wait();
			delete p->runner;
			p->runner = NULL;
		}
	}
}

void FalseColorModel::createRunner(coloring type)
{
	payload *p = map.value(type);
	assert(p != NULL && p->runner == NULL);
	// runners are only deleted in reset(). if reset() was not called and the
	// runner exists, the image has been calculated already, so there is no
	// reason to call craeteRunner

	// init runner & command
	p->runner = new CommandRunner();
	std::map<std::string, boost::any> input;
	input["multi_img"] = shared_img;
	p->runner->input = input;
	p->runner->cmd = new gerbil::RGB(); // deleted in ~CommandRunner()

	// TODO: init rgb.config, if non-default setup is neccessary. then we need
	// a copy of the initialized RGB object, as the obj in the runner is
	// deleted in its destructor
	gerbil::RGB *cmd = (gerbil::RGB *)p->runner->cmd;
	switch (type)
	{
	case CMF:
		cmd->config.algo = gerbil::COLOR_XYZ;
		break;
	case PCA:
		cmd->config.algo = gerbil::COLOR_PCA;
		break;
#ifndef WITH_EDGE_DETECT
	case SOM:
		cmd->config.algo = gerbil::COLOR_SOM;
		break;
#endif
	default:
		// coloring type is COLSIZE, SOM without edge detect or missing
		assert(false);
	}

	QObject::connect(
		this, SIGNAL(terminateRunners()),
		p->runner, SLOT(terminate()), Qt::QueuedConnection);
	QObject::connect(
		p->runner, SIGNAL(success(std::map<std::string, boost::any>)),
		p, SLOT(propagateRunnerSuccess(std::map<std::string, boost::any>)),
		Qt::QueuedConnection);
}


void FalseColorModel::calculateForeground(coloring type)
{
	payload *p = map.value(type);
	assert(p != NULL);

	// img is calculated already
	if (!p->img.isNull()) {
		emit calculationComplete(type, p->img);
		return;
	}

	// img is currently in calculation, loadComplete will be emitted as soon as its finished
	if (p->calcInProgress)
		return;

	// we can't get around doing some real calculations
	p->calcInProgress = true;
	if (type == CMF) {
		BackgroundTaskPtr taskRgb(new RgbTbb(
			shared_img,
			mat3f_ptr(new SharedData<cv::Mat3f>(new cv::Mat3f)),
			p->calcImg));
		taskRgb.get()->run();
	}
	else {
		createRunner(type);
		cv::Mat3b result;
		{
			SharedMultiImgBaseGuard guard(*shared_img);
			gerbil::RGB *cmd = (gerbil::RGB *)p->runner->cmd;
			result = (cv::Mat3b)(cmd->execute(**shared_img) * 255.0f);
		}
		p->img = vole::Mat2QImage(result);
	}
	p->calcInProgress = false;
	emit calculationComplete(type, p->img);
}

void FalseColorModel::calculateBackground(coloring type)
{
	payload *p = map.value(type);
	assert(p != NULL);

	// img is calculated already
	if (!p->img.isNull()) {
		emit calculationComplete(type, p->img);
		return;
	}

	// img is currently in calculation, loadComplete will
	// be emitted as soon as its finished
	if (p->calcInProgress)
		return;

	// we can't get around doing some real calculations
	p->calcInProgress = true;
	if (type == CMF) {
		BackgroundTaskPtr taskRgb(new RgbTbb(
			shared_img,
			mat3f_ptr(new SharedData<cv::Mat3f>(new cv::Mat3f)),
			p->calcImg));
		QObject::connect(taskRgb.get(), SIGNAL(finished(bool)),
						 p, SLOT(propagateFinishedQueueTask(bool)),
						 Qt::QueuedConnection);
		queue->push(taskRgb);
	}
	else {
		createRunner(type);
		/* note that this is an operation that cannot run concurrently with
		 * other tasks that would invalidate (swap) the image data.
		 * Make sure to cancel this before enqueueing tasks that invalidate
		 * the image data! */
		p->runner->start();
	}
}

void FalseColorModelPayload::propagateFinishedQueueTask(bool success)
{
	calcInProgress = false;

	if (!success)
		return;

	img = **calcImg;

	emit calculationComplete(type, img);
}

void FalseColorModelPayload::propagateRunnerSuccess(std::map<std::string, boost::any> output)
{
	/* ensure that we did not cancel computation (and delete the worker) */
	if (!runner)
		return;

	img = boost::any_cast<QImage>(output["multi_img"]);

	calcInProgress = false;
	emit calculationComplete(type, img);
}
