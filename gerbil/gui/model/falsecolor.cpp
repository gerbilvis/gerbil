#include "falsecolor.h"

#include "representation.h"
#include "../commandrunner.h"
#include "../tasks/rgbtbb.h"

#include <shared_data.h>
#include <background_task/background_task_queue.h>
#include <rgb.h>
#include <multi_img.h>
#include <qtopencv.h>

#include <QImage>
#include <QPixmap>
#include <opencv2/core/core.hpp>

// TODO RGB:
//  progressUpdate() in SOM einbauen,
//      SOM *som = SOMTrainer::train(config.som, img);

FalseColorModel::FalseColorModel(QObject *parent, BackgroundTaskQueue *queue)
	: QObject(parent), queue(queue)
{
	int type = QMetaType::type("coloring");
	if (type == 0 || !QMetaType::isRegistered(type))
		qRegisterMetaType<coloring>("coloring");

	type = QMetaType::type("std::map<std::string, boost::any>");
	if (type == 0 || !QMetaType::isRegistered(type))
		qRegisterMetaType< std::map<std::string, boost::any> >(
					"std::map<std::string, boost::any>");

	for (int gradient = 0; gradient < 2; ++gradient)
	{
		bool grad = gradient == 1; // true in first iteration, false in second

		for (int i = 0; i < COLSIZE; ++i) {
#ifndef WITH_EDGE_DETECT
			if (i == SOM)
				continue;
#endif
			coloringWithGrad mapId;
			mapId.col = (coloring)i;
			mapId.grad = grad;

			payload *p = new payload(representation::IMG, (coloring)i, grad);
			map.insert(mapId, p);

			QObject::connect(
				p,    SIGNAL(calculationComplete(coloring, bool, QPixmap)),
				this, SIGNAL(calculationComplete(coloring, bool, QPixmap)),
				Qt::QueuedConnection);
		}
	}
}

FalseColorModel::~FalseColorModel()
{
	cancel();
}

void FalseColorModel::setMultiImg(representation::t type,
								  SharedMultiImgPtr shared_img)
{
	// in the future, we might be interested in the other ones as well.
	// currently, we don't process other types, so "warn" the caller
	assert(type == representation::IMG || type == representation::GRAD);

	if (type == representation::IMG)
		this->shared_img = shared_img;
	else if (type == representation::GRAD)
		this->shared_grad = shared_img;

	reset();
}

void FalseColorModel::processImageUpdate(representation::t type,
										 SharedMultiImgPtr img)
{
	if (type == representation::IMG || type == representation::GRAD)
		reset();
}

void FalseColorModel::reset()
{
	// in case we have some computation going on
	cancel();

	// reset all images
	PayloadList l = map.values();
	foreach(payload *p, l) {
		p->img = QPixmap();
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

void FalseColorModel::createRunner(coloringWithGrad mapId)
{
	payload *p = map.value(mapId);
	assert(p != NULL && p->runner == NULL);
	// runners are only deleted in reset(). if reset() was not called and the
	// runner exists, the image has been calculated already, so there is no
	// reason to call craeteRunner

	// init runner & command
	p->runner = new CommandRunner();
	std::map<std::string, boost::any> input;
	input["multi_img"] = mapId.grad ? shared_grad : shared_img;
	p->runner->input = input;
	p->runner->cmd = new gerbil::RGB(); // deleted in ~CommandRunner()

	// TODO: init rgb.config, if non-default setup is neccessary. then we need
	// a copy of the initialized RGB object, as the obj in the runner is
	// deleted in its destructor
	gerbil::RGB *cmd = (gerbil::RGB *)p->runner->cmd;
	switch (mapId.col)
	{
	case CMF:
		cmd->config.algo = gerbil::COLOR_XYZ;
		break;
	case PCA:
		cmd->config.algo = gerbil::COLOR_PCA;
		break;
#ifdef WITH_EDGE_DETECT
	case SOM:
		cmd->config.algo = gerbil::COLOR_SOM;

		// default parameters for false coloring (different to regular defaults)

		// CONE parameters
//		cmd->config.som.type		= vole::SOM_CONE;
//		cmd->config.som.granularity	= 0.06; // 1081 neurons
//		cmd->config.som.sigmaStart  = 0.12;
//		cmd->config.som.sigmaEnd    = 0.03;
//		cmd->config.som.learnStart  = 0.75;
//		cmd->config.som.learnEnd    = 0.01;

		// CUBE parameters
		cmd->config.som.type        = vole::SOM_CUBE;
		cmd->config.som.sidelength  = 10;
		cmd->config.som.sigmaStart  = 4;
		cmd->config.som.sigmaEnd    = 1;
		cmd->config.som.learnStart  = 0.75;
		cmd->config.som.learnEnd    = 0.01;

		cmd->config.som.maxIter     = 100000;
		// TODO: should som really have its own verbosity setting?
		cmd->config.som.verbosity = 3; // TODO: delete
		cmd->config.som.seed = 0; // TODO: delete
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


void FalseColorModel::computeForeground(coloring type, bool gradient)
{
	coloringWithGrad mapId;
	mapId.col = type;
	mapId.grad = gradient;

	payload *p = map.value(mapId);
	assert(p != NULL);

	// img is calculated already
	if (!p->img.isNull()) {
		emit calculationComplete(type, gradient, p->img);
		return;
	}

	// img is currently in calculation, loadComplete will be emitted as soon as its finished
	if (p->calcInProgress)
		return;

	// we can't get around doing some real calculations
	p->calcInProgress = true;
	if (type == CMF) {
		BackgroundTaskPtr taskRgb(new RgbTbb(
			gradient ? shared_grad : shared_img,
			mat3f_ptr(new SharedData<cv::Mat3f>(new cv::Mat3f)),
			p->calcImg));
		taskRgb.get()->run();
	}
	else {
		createRunner(mapId);
		cv::Mat3b result;
		{
			SharedMultiImgBaseGuard guard(gradient ? *shared_grad : *shared_img);
			gerbil::RGB *cmd = (gerbil::RGB *)p->runner->cmd;
			result = (cv::Mat3b)(cmd->execute(**shared_img) * 255.0f);
		}
		p->img.convertFromImage(vole::Mat2QImage(result));
	}
	p->calcInProgress = false;
	emit calculationComplete(type, gradient, p->img);
}

void FalseColorModel::computeBackground(coloring type, bool gradient)
{
	coloringWithGrad mapId;
	mapId.col = type;
	mapId.grad = gradient;

	payload *p = map.value(mapId);
	assert(p != NULL);

	// img is calculated already
	if (!p->img.isNull()) {
		emit calculationComplete(type, gradient, p->img);
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
			gradient ? shared_grad : shared_img,
			mat3f_ptr(new SharedData<cv::Mat3f>(new cv::Mat3f)),
			p->calcImg));
		QObject::connect(taskRgb.get(), SIGNAL(finished(bool)),
						 p, SLOT(propagateFinishedQueueTask(bool)),
						 Qt::QueuedConnection);
		queue->push(taskRgb);
	}
	else {
		createRunner(mapId);
		/* note that this is an operation that cannot run concurrently with
		 * other tasks that would invalidate (swap) the image data.
		 * Make sure to cancel this before enqueueing tasks that invalidate
		 * the image data! */
		p->runner->start();
	}
}

void FalseColorModel::returnIfCached(coloring type, bool gradient)
{
	coloringWithGrad mapId;
	mapId.col = type;
	mapId.grad = gradient;

	payload *p = map.value(mapId);
	assert(p != NULL);

	// img is calculated already
	if (!p->img.isNull()) {
		emit calculationComplete(type, gradient, p->img);
	}
}

void FalseColorModelPayload::propagateFinishedQueueTask(bool success)
{
	calcInProgress = false;

	if (!success)
		return;

	img.convertFromImage(**calcImg);

	emit calculationComplete(type, gradient, img);
}

void FalseColorModelPayload::propagateRunnerSuccess(std::map<std::string, boost::any> output)
{
	/* ensure that we did not cancel computation (and delete the worker) */
	if (!runner)
		return;

	cv::Mat3f mat = boost::any_cast<cv::Mat3f>(output["multi_img"]);
	img.convertFromImage(vole::Mat2QImage((cv::Mat3b)mat));

	calcInProgress = false;
	emit calculationComplete(type, gradient, img);
}
