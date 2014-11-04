#include "clusteringmodel.h"

#include "../commandrunner.h"
#include <meanshift_shell.h>

#include <boost/any.hpp>

#define GGDBG_MODULE
#include "gerbil_gui_debug.h"

ClusteringModel::ClusteringModel()
	: commandRunner(NULL),
	  state(State::Idle)
{
	// for CommandRunner result slot, onSegmentationCompleted
	qRegisterMetaType< std::map<std::string, boost::any> >(
				"std::map<std::string, boost::any>");
}

ClusteringModel::~ClusteringModel()
{
	if (NULL != commandRunner) {
		commandRunner->terminate();
		commandRunner->wait();
		commandRunner->deleteLater();
	}
}

void ClusteringModel::requestSegmentation(
		shell::Command* cmd, int numbands, bool gradient)
{
	// abort currently executing CommandRunner, if any
	abortCommandRunner();

	if (request) {
		std::cerr << "ClusteringModel::requestSegmentation(): "
				  << "bad state, request != NULL" << std::endl;
		return;
	}
	request = boost::shared_ptr<Request>(new Request());
	request->cmd = cmd;
	request->numbands = numbands;
	request->gradient = gradient;

	if (State::Idle == state) {
		// We are not subscribed for image data yet. Subscribe and wait.
		state = State::Subscribed;
		if (!gradient) {
			GGDBGM("subscribing for " << representation::IMG << endl);
			emit subscribeRepresentation(this, representation::IMG);
		} else {
			// This will be the good behaviour
//			GGDBGM("subscribing for " << representation::GRAD << endl);
//			emit subscribeRepresentation(this, representation::GRAD);

			// HACK, FIXME:This is the old BAD behaviour
			GGDBGM("representation::GRAD HACK in effect" << endl);
			startSegmentation();
		}
	} else {
		GGDBGM("we are non-Idle, starting segmentation right away" << endl);
		startSegmentation();
	}
}

void ClusteringModel::startSegmentation()
{
	bool good = true;
	if (!request) {
		std::cerr << "ClusteringModel::startSegmentation(): "
				  << "request is NULL" << std::endl;
		good = false;
	}

	{
		SharedMultiImgBaseGuard guard(*image);
		if ((*image)->empty()) {
			std::cerr << "ClusteringModel::startSegmentation(): "
					  << "image is empty" << std::endl;
			good = false;
		}
	}

	if (!good) {
		return;
	}

	GGDBGM("kicking off computation" << endl);

	commandRunner = new CommandRunner();
	commandRunner->cmd = request->cmd;

	connect(commandRunner, SIGNAL(progressChanged(int)),
			this, SIGNAL(progressChanged(int)));

	connect(commandRunner,
			SIGNAL(success(std::map<std::string,boost::any>)),
			this,
			SLOT(processSegmentationCompleted(
					 std::map<std::string,boost::any>)));

	// create a copy of the image
	boost::shared_ptr<multi_img> input, inputgrad;
	{
		SharedMultiImgBaseGuard guard(*image);
		input = boost::shared_ptr<multi_img>(new multi_img(**image));
	}

	if (request->numbands > 0 && request->numbands < (int) input->size()) {
		boost::shared_ptr<multi_img> input_tmp(
					new multi_img(input->spec_rescale(request->numbands)));
		input = input_tmp;
	}

	seg_meanshift::MeanShiftConfig &config =
			static_cast<seg_meanshift::MeanShiftShell*>(request->cmd)->config;
	if (request->gradient) {
		// copy needed here (TODO: only in sp_withGrad case)
		multi_img loginput(*input, true);
		loginput.apply_logarithm();

		// FIXME: Er, shouldn't we use the GRAD image supplied by ImageModel?
		if (config.sp_withGrad) {
			// use gradient only as second argument
			inputgrad = boost::shared_ptr<multi_img>(
						new multi_img(loginput.spec_gradient()));
		} else {
			// method expects gradient as input (we can free the original data)
			input = boost::shared_ptr<multi_img>(new multi_img(
													 loginput.spec_gradient()));
		}
	}

	commandRunner->input["multi_img"] = input;
	commandRunner->input["multi_grad"] = inputgrad;

	GGDBGM("CommandRunner object is "
		   <<  static_cast<CommandRunner*>(commandRunner) << endl);
	commandRunner->start();
	state = State::Executing;
}


void ClusteringModel::cancel()
{
	abortCommandRunner();
	request = boost::shared_ptr<Request>();
}

void ClusteringModel::processImageUpdate(representation::t repr,
										 SharedMultiImgPtr image,
										 bool duplicate)
{
	// FIXME, see below
	bool unsafe = State::Subscribed != state;

	if (State::Idle == state) {
		GGDBGM("we are in Idle state" << endl);
		return;
	}

	if (State::Executing == state && duplicate) {
		// If we are executing, we only restart on new image data,
		// i.e. duplicate == false.
		GGDBGM("duplicate update" << endl);
		return;
	}

#ifdef WITH_IMGNORM // HACK: we prefer the normed version if we have it.
	if (repr != representation::NORM)
#else
	if (repr != representation::IMG)
#endif
	{
		GGDBGM("we are not interested in " << repr <<  endl);
		return;
	}
	this->image = image;

	// Tell running CommandRunner to abort.
	if (commandRunner) {
		GGDBGM("canceling running segmentation"<< endl);
		abortCommandRunner();
		// reset GUI
		emit progressChanged(100);
		emit segmentationCompleted();
	}

	// FIXME need to fix Request containing Command pointer...
	if (unsafe) {
		// We can't start the same Command object more than once!
		GGDBGM("we are non-Idle, and would start segmentation right away "
			   "with new data, but the implementation is still too buggy "
			   "to re-run an already started jobs. Sorry!" << endl);
		request = boost::shared_ptr<Request>();
		state = State::Idle;
		GGDBGM("unsubscribing representations" << endl);
		emit unsubscribeRepresentation(this, representation::IMG);
		#ifdef WITH_IMGNORM
		emit unsubscribeRepresentation(this, representation::NORM);
		#endif
		emit unsubscribeRepresentation(this, representation::GRAD);
		return;
	}

	startSegmentation();
}

void ClusteringModel::processSegmentationCompleted(
		std::map<std::string, boost::any> output)
{
	state = State::Idle;
	request = boost::shared_ptr<Request>();

	// We are back to Idle, just unsubscribe everything.
	GGDBGM("unsubscribing representations" << endl);
	emit unsubscribeRepresentation(this, representation::IMG);
	#ifdef WITH_IMGNORM
	emit unsubscribeRepresentation(this, representation::NORM);
	#endif
	emit unsubscribeRepresentation(this, representation::GRAD);
	if (output.count("labels")) {
		boost::shared_ptr<cv::Mat1s> labelMask =
				boost::any_cast< boost::shared_ptr<cv::Mat1s> >(
					output["labels"]);
		// was: setLabels(*labelMask);
		emit setLabelsRequested(*labelMask);
	}

	if (output.count("findKL.K") && output.count("findKL.L")) {
		int foundK = boost::any_cast<int>(output["findKL.K"]);
		int foundL = boost::any_cast<int>(output["findKL.L"]);
		emit resultKL(foundK, foundL);
	}
	emit segmentationCompleted();
}

void ClusteringModel::abortCommandRunner()
{
	// FIXME: the Meanshift will not abort, but continue running
	// in the background. Resource starvation...

	if(NULL == commandRunner)
		return;
	// disconnect all signals
	disconnect(commandRunner, 0, 0, 0);
	// note: CommandRunner overrides terminate(), it just cancels.
	commandRunner->terminate();
	// make sure the old runner is deleted after the thread has joined.
	connect(commandRunner, SIGNAL(finished()),
			commandRunner, SLOT(deleteLater()));
	commandRunner = NULL;
}
