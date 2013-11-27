#include "clusteringmodel.h"

#include "../commandrunner.h"
#include <meanshift_shell.h>

#include <boost/any.hpp>

#include "gerbil_gui_debug.h"

ClusteringModel::ClusteringModel()
	: cmdr(NULL)
{
}

ClusteringModel::~ClusteringModel()
{
	if(NULL != cmdr) {
		cmdr->terminate();
		cmdr->wait();
		delete cmdr;
	}
}

void ClusteringModel::setMultiImage(SharedMultiImgPtr image)
{
	this->image = image;
}

void ClusteringModel::startSegmentation(
		vole::Command* cmd, int numbands, bool gradient)
{
	// get rid of previous command runner  (HACK)
	if(NULL != cmdr) {
		// FIXME: the Meanshift will not abort, but continue running
		// in the background. Resource starvation...
		silenceCommandRunner();
	}

	cmdr = new CommandRunner();
	cmdr->cmd = cmd;

	connect(cmdr, SIGNAL(progressChanged(int)),
			this, SIGNAL(progressChanged(int)));
	qRegisterMetaType< std::map<std::string, boost::any> >(
				"std::map<std::string, boost::any>");
	connect(cmdr, SIGNAL(success(std::map<std::string,boost::any>)),
			this, SLOT(onSegmentationCompleted(std::map<std::string,boost::any>)));

	boost::shared_ptr<multi_img> input, inputgrad;
	{
		SharedMultiImgBaseGuard guard(*image);
		input = boost::shared_ptr<multi_img>(new multi_img(**image));
	}

	if (numbands > 0 && numbands < (int) input->size()) {
		boost::shared_ptr<multi_img> input_tmp(new multi_img(input->spec_rescale(numbands)));
		input = input_tmp;
	}

	vole::MeanShiftConfig &config =
			static_cast<vole::MeanShiftShell*>(cmd)->config;
	if (gradient) {
		// copy needed here (TODO: only in sp_withGrad case)
		multi_img loginput(*input);
		loginput.apply_logarithm();

		if (config.sp_withGrad) {
			// use gradient only as second argument
			inputgrad = boost::shared_ptr<multi_img>(new multi_img(loginput.spec_gradient()));
		} else {
			// method expects gradient as input (we can free the original data)
			input = boost::shared_ptr<multi_img>(new multi_img(loginput.spec_gradient()));
		}
	}

	cmdr->input["multi_img"] = input;
	cmdr->input["multi_grad"] = inputgrad;

	//GGDBGM("CommandRunner object is "<< static_cast<CommandRunner*>(cmdr) << endl);
	cmdr->start();
}


void ClusteringModel::cancel()
{
	cmdr->terminate();
}

void ClusteringModel::processImageUpdate(representation::t type, SharedMultiImgPtr image)
{
	//GGDBGM("type " << type << endl);
	if(representation::IMG == type && NULL != cmdr) {
		//GGDBGM("canceling running segmentation"<< endl);
		// Get rid of currently running CommandRunner.
		silenceCommandRunner();
		// reset GUI
		emit progressChanged(100);
		emit segmentationCompleted();
	}
}

void ClusteringModel::onSegmentationCompleted(
		std::map<std::string, boost::any> output)
{
	if (output.count("labels")) {
		boost::shared_ptr<cv::Mat1s> labelMask =
				boost::any_cast< boost::shared_ptr<cv::Mat1s> >(output["labels"]);
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

void ClusteringModel::silenceCommandRunner()
{
	if(NULL == cmdr)
		return;
	// disconnect all signals
	disconnect(cmdr, 0, 0, 0);
	// note: CommandRunner overrides terminate(), it just cancels.
	cmdr->terminate();
	// make sure the old runner is deleted after the thread has joined.
	connect(cmdr, SIGNAL(finished()), cmdr, SLOT(deleteLater()));
	cmdr = NULL;
}
