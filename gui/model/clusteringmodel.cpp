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
		shell::Command* cmd, bool gradient)
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
	request->gradient = gradient;
	if (gradient) {
		request->repr = representation::GRAD;
	} else {
		request->repr = representation::NORM;
	}

	if (State::Idle == state) {
		// We are not subscribed for image data yet.
		state = State::Subscribed;

		// Unsubscribe other stale subscription, if any.
		emit unsubscribeRepresentation(this, representation::NORM);
		emit unsubscribeRepresentation(this, representation::GRAD);

		// reset input pointers
		inputMap[representation::NORM] = SharedMultiImgPtr();
		inputMap[representation::GRAD] = SharedMultiImgPtr();

		// Subscribe and wait.
		GGDBGM("subscribing " << endl);
		emit subscribeRepresentation(this, representation::NORM);
		emit subscribeRepresentation(this, representation::GRAD);
	} else {
		GGDBGM("we are non-Idle, (re-)starting segmentation right away" << endl);
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

	QList<representation::t> reps;
	reps.append(representation::NORM);
	reps.append(representation::GRAD);
	foreach (representation::t repr, reps) {
		if (!inputMap[repr]) {
			std::cerr << "ClusteringModel::startSegmentation(): "
					  << "input " << request->repr << " is NULL"
					  << std::endl;
			good = false;
		} else {
			SharedMultiImgBaseGuard guard(*inputMap[repr]);
			if ((*inputMap[repr])->empty()) {
				std::cerr << "ClusteringModel::startSegmentation(): "
						  << "input " << request->repr << " is empty"
						  << std::endl;
				good = false;
			}
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
	connect(commandRunner,
			SIGNAL(failure()),
			this,
			SLOT(processSegmentationFailed()));

	// create copies of the input image images
	{
		// Meanshift always needs the IMG/NORM representation for SUPERPIXEL.
		SharedMultiImgBaseGuard guard(*inputMap[representation::NORM]);
		commandRunner->input["multi_img"] =
				boost::shared_ptr<multi_img>(
					new multi_img(**inputMap[representation::NORM]));
	}
	if (representation::NORM == request->repr) {
		commandRunner->input["multi_grad"] = boost::shared_ptr<multi_img>();
	} else if (representation::GRAD == request->repr) {
		SharedMultiImgBaseGuard guard(*inputMap[representation::GRAD]);
		commandRunner->input["multi_grad"] =
						boost::shared_ptr<multi_img>(
							new multi_img(**inputMap[representation::GRAD]));;
	} else {
		std::cerr << "ClusteringModel::startSegmentation(): "
				  << "bad representation in request: "
				  << request->repr << std::endl;
		delete commandRunner;
		return;
	}

	GGDBGM("CommandRunner object is "
		   <<  static_cast<CommandRunner*>(commandRunner) << endl);
	commandRunner->start();
	state = State::Executing;
}


void ClusteringModel::cancel()
{
	abortCommandRunner();
	// CommandRunner may not send failure or complete signal after abort which
	// would leave us in executing state forever.
	resetToIdle();
}

void ClusteringModel::resetToIdle()
{
	GGDBGM("deleteing request, unsubscribing representations, state = Idle"
		   << endl);
	request = boost::shared_ptr<Request>();
	state = State::Idle;
	emit unsubscribeRepresentation(this, representation::NORM);
	emit unsubscribeRepresentation(this, representation::GRAD);
}

void ClusteringModel::processImageUpdate(representation::t repr,
										 SharedMultiImgPtr image,
										 bool duplicate)
{
	// FIXME, see below
	bool unsafe = State::Subscribed != state;

	if (State::Idle == state) {
		if (!duplicate) {
			GGDBGM("we are in Idle state" << endl);
		}
		return;
	} else if (State::Executing == state && duplicate) {
		// If we are executing, we only restart on new image data,
		// i.e. duplicate == false.
		GGDBGM("duplicate update" << endl);
		return;
	} else if (State::Executing == state &&
			   !duplicate &&
			   inputMap[representation::NORM] &&
			   inputMap[representation::GRAD])
	{
		GGDBGM("update while executing, reseting input" << endl);
		// reset input pointers
		inputMap[representation::NORM] = SharedMultiImgPtr();
		inputMap[representation::GRAD] = SharedMultiImgPtr();
	}

	if (!request) {
		std::cerr << "ClusteringModel::processImageUpdate(): "
				  << "request is NULL" << std::endl;
		return;
	}

	if (representation::NORM == repr || representation::GRAD == repr)
	{
		GGDBGM("saving pointer to " << repr <<  endl);
	} else {
		GGDBGM("we are not interested in " << repr <<  endl);
		return;
	}

	inputMap[repr] = image;

	if (!(inputMap[representation::NORM] && inputMap[representation::GRAD])) {
		GGDBGM("input not yet complete" << endl);
		return;
	}

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
		resetToIdle();
		return;
	}

	startSegmentation();
}

void ClusteringModel::processSegmentationCompleted(
		std::map<std::string, boost::any> output)
{
	resetToIdle();
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

void ClusteringModel::processSegmentationFailed()
{
	resetToIdle();
}

void ClusteringModel::abortCommandRunner()
{
	// FIXME: the Meanshift will not abort, but continue running
	// in the background. Resource starvation...

	if (NULL == commandRunner) {
		return;
	}
	// disconnect all signals
	disconnect(commandRunner, 0, 0, 0);
	// note: CommandRunner overrides terminate(), it just cancels.
	commandRunner->terminate();
	// make sure the old runner is deleted after the thread has joined.
	connect(commandRunner, SIGNAL(finished()),
			commandRunner, SLOT(deleteLater()));
	commandRunner = NULL;
}
