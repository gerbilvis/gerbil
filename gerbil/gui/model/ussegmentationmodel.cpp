#include "ussegmentationmodel.h"

#include "../commandrunner.h"

#include <boost/any.hpp>


UsSegmentationModel::UsSegmentationModel(QObject *parent) :
	QObject(parent), cmdr(NULL)
{
}

UsSegmentationModel::~UsSegmentationModel()
{
	if(NULL != cmdr) {
		cmdr->terminate();
		cmdr->wait();
		delete cmdr;
	}
}

void UsSegmentationModel::setMultiImage(SharedMultiImgPtr image)
{
	this->image = image;
}

void UsSegmentationModel::startSegmentation(
		vole::Command* cmd, int numbands, bool gradient)
{
	// wait for previous command runner to terminate
	if(NULL != cmdr) {
		cmdr->wait();
		delete cmdr;
	}

	cmdr = new CommandRunner();
	cmdr->cmd = cmd;

	connect(cmdr, SIGNAL(progressChanged(int)),
			this, SIGNAL(progressChanged(int)));
	qRegisterMetaType< std::map<std::string, boost::any> >(
				"std::map<std::string, boost::any>");
	connect(cmdr, SIGNAL(success(std::map<std::string,boost::any>)),
			this, SLOT(onSegmentationCompleted(std::map<std::string,boost::any>)));

	boost::shared_ptr<multi_img> input;
	{
		SharedMultiImgBaseGuard guard(*image);
		input = boost::shared_ptr<multi_img>(new multi_img(**image));
	}

	if (numbands > 0 && numbands < (int) input->size()) {
		boost::shared_ptr<multi_img> input_tmp(new multi_img(input->spec_rescale(numbands)));
		input = input_tmp;
	}

	// 2013-06-17 altmann:
	// actually image currently is always IMG, so the gradient flag probably
	// doesn't make much sense.
	if (gradient) {
		// copy needed here
		multi_img loginput(*input);
		loginput.apply_logarithm();
		input = boost::shared_ptr<multi_img>(new multi_img(loginput.spec_gradient()));
	}

	cmdr->input["multi_img"] = input;

	cmdr->start();
}


void UsSegmentationModel::cancel()
{
	cmdr->terminate();
}

void UsSegmentationModel::onSegmentationCompleted(
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
