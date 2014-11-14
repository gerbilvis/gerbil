#include "falsecolormodelpayload.h"
#include "commandrunner.h"
#include "rgb.h"
#include "qtopencv.h"

#include <som_config.h>

void FalseColorModelPayload::run()
{
	runner = new CommandRunner();

	std::map<std::string, boost::any> input;
	if(FalseColoring::isBasedOn(coloringType, representation::IMG)) {
		input["multi_img"] = img;
	} else {
		input["multi_img"] = grad;
	}
	runner->input = input;
	rgb::RGBDisplay *cmd = new rgb::RGBDisplay(); // object owned by CommandRunner

	switch (coloringType)
	{
	case FalseColoring::CMF:
		cmd->config.algo = rgb::COLOR_XYZ;
		break;
	case FalseColoring::PCA:
	case FalseColoring::PCAGRAD:
		cmd->config.algo = rgb::COLOR_PCA;
		break;
#ifdef WITH_SOM
	case FalseColoring::SOM:
	case FalseColoring::SOMGRAD:
		// default parameters for false coloring (different to regular defaults)
		cmd->config.algo = rgb::COLOR_SOM;
		cmd->config.som.maxIter = 50000;
		cmd->config.som.seed = time(NULL);

		// CUBE parameters
		cmd->config.som.type        = som::SOM_CUBE;
		cmd->config.som.dsize       = 10;
		cmd->config.som.sigmaStart  = 4;
		cmd->config.som.sigmaEnd    = 1;
		cmd->config.som.learnStart  = 0.75;
		cmd->config.som.learnEnd    = 0.01;

		break;
#endif /* WITH_SOM */
	default:
		assert(false);
	}
	runner->setCommand(cmd);
	connect(runner, SIGNAL(success(std::map<std::string, boost::any>)),
			this, SLOT(processRunnerSuccess(std::map<std::string, boost::any>)));
	connect(runner, SIGNAL(failure()),
			this, SLOT(processRunnerFailure()));
	connect(runner, SIGNAL(progressChanged(int)),
			this, SLOT(processRunnerProgress(int)));
	// start thread
	runner->start();
}

void FalseColorModelPayload::cancel()
{
	//GGDBGM( coloringType << endl);
	canceled = true;
	if (runner) {
		runner->abort();
	}
}

void FalseColorModelPayload::processRunnerProgress(int percent)
{
	if (canceled) {
		return;
	}
	emit progressChanged(coloringType, percent);
}


void FalseColorModelPayload::processRunnerSuccess(std::map<std::string, boost::any> output)
{
	runner->deleteLater();
	if(canceled) {
		return;
	}
	cv::Mat3f mat = boost::any_cast<cv::Mat3f>(output["multi_img"]);
	result.convertFromImage(Mat2QImage((cv::Mat3b)mat));
	emit finished(coloringType, true); // success
}

void FalseColorModelPayload::processRunnerFailure()
{
	runner->deleteLater();
	emit finished(coloringType, false); // failure
}
