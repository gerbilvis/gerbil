#include <QImage>
#include <QPixmap>
#include <opencv2/core/core.hpp>

#include <shared_data.h>
#include <rgb.h>
#include <multi_img.h>
#include <qtopencv.h>

#include "representation.h"
#include "../commandrunner.h"

#include "falsecolormodel.h"

#include <gerbil_gui_debug.h>

QList<FalseColoring::Type> FalseColoring::allList = QList<FalseColoring::Type>()
	<< FalseColoring::CMF
	<< FalseColoring::CMFGRAD
	<< FalseColoring::PCA
	<< FalseColoring::PCAGRAD
	<< FalseColoring::SOM
	<< FalseColoring::SOMGRAD;

std::ostream &operator<<(std::ostream& os, const FalseColoring::Type& coloringType)
{
	if (coloringType < 0 ||
			coloringType >= FalseColoring::Type(FalseColoring::SIZE)) {
		os << "INVALID";
		return os;
	}
	const char * const str[] = { "CMF", "CMFGRAD", "PCA", "PCAGRAD", "SOM", "SOMGRAD" };
	os << str[coloringType];
	return os;
}


FalseColorModel::FalseColorModel()
{
	int type = QMetaType::type("FalseColoring");
	if (type == 0 || !QMetaType::isRegistered(type))
		qRegisterMetaType<FalseColoring::Type>("FalseColoring");

	type = QMetaType::type("std::map<std::string, boost::any>");
	if (type == 0 || !QMetaType::isRegistered(type))
		qRegisterMetaType< std::map<std::string, boost::any> >(
					"std::map<std::string, boost::any>");
}

FalseColorModel::~FalseColorModel()
{
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

	cache.clear();
}

void FalseColorModel::processImageUpdate(representation::t type,
										 SharedMultiImgPtr)
{

	// make sure no computations based on old image data make it into the
	// cache
	FalseColorModelPayloadMap::iterator payloadIt;
	for(payloadIt = payloads.begin(); payloadIt != payloads.end(); payloadIt++) {
		FalseColoring::Type coloringType = payloadIt.key();
		if(FalseColoring::isBasedOn(coloringType, type)) {
			//GGDBGM("canceling " << coloringType << endl);
			cancelComputation(coloringType);
		}
	}

	QList<FalseColoring::Type> outOfDateList;
	outOfDateList.reserve(FalseColoring::size());

	// invalidate affected cache entries:
	FalseColoringCache::iterator it;
	for(it=cache.begin(); it != cache.end(); it++) {
		FalseColoring::Type coloringType = it.key();
		if(FalseColoring::isBasedOn(coloringType, type)) {
			it.value().upToDate = false;
			outOfDateList.append(coloringType);
		}
	}
	foreach(FalseColoring::Type coloringType, outOfDateList) {
		emit coloringOutOfDate(coloringType);
	}
}

void FalseColorModel::requestColoring(FalseColoring::Type coloringType, bool recalc)
{
	//GGDBG_CALL();
	FalseColoringCache::iterator cacheIt = cache.find(coloringType);
	if(cacheIt != cache.end() && cacheIt->upToDate) {
		if(recalc &&
				// recalc makes sense only for SOM, the other representations
				// are deterministic -> no need to recompute
				!FalseColoring::isDeterministic(coloringType))
		{
			computeColoring(coloringType);
		} else {
			emit coloringComputed(coloringType, cacheIt->img);
		}
	} else {
		computeColoring(coloringType);
	}
}

void FalseColorModel::computeColoring(FalseColoring::Type coloringType)
{
	FalseColorModelPayloadMap::iterator payloadIt = payloads.find(coloringType);
	if(payloadIt != payloads.end()) {
		// computation in progress
		return;
	}
	FalseColorModelPayload *payload =
			new FalseColorModelPayload(coloringType, shared_img, shared_grad);
	payloads.insert(coloringType, payload);
	connect(payload, SIGNAL(finished(FalseColoring::Type, bool)),
			this, SLOT(processComputationFinished(FalseColoring::Type, bool)));
	// forward progress signal
	connect(payload, SIGNAL(progressChanged(FalseColoring::Type,int)),
			this, SIGNAL(progressChanged(FalseColoring::Type,int)));
	payload->run();
}

void FalseColorModel::cancelComputation(FalseColoring::Type coloringType)
{
	//GGDBGM(coloringType<<endl);
	FalseColorModelPayload *payload = NULL;
	FalseColorModelPayloadMap::iterator payloadIt = payloads.find(coloringType);
	if(payloadIt != payloads.end()) {
		//GGDBGM("canceling "<< coloringType << endl);
		payload = *payloadIt;
		assert(payload);
		// set flag
		payload->cancel();
	}
}

void FalseColorModel::processComputationFinished(FalseColoring::Type coloringType, bool success)
{
	QPixmap pixmap;
	FalseColorModelPayload *payload;
	FalseColorModelPayloadMap::iterator payloadIt = payloads.find(coloringType);
	assert(payloadIt != payloads.end());
	payload = *payloadIt;
	assert(payload);
	payloads.erase(payloadIt);
	if(success) {
		pixmap = payload->getResult();
		cache.insert(coloringType,FalseColoringCacheItem(pixmap));
	}
	payload->deleteLater();
	if(success) {
		emit coloringComputed(coloringType, pixmap);
	} else {
		//GGDBGM("emitting computationCancelled " << coloringType<<endl);
		emit computationCancelled(coloringType);
	}
}


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
	gerbil::RGB *cmd = new gerbil::RGB(); // object owned by CommandRunner

	switch (coloringType)
	{
	case FalseColoring::CMF:
	case FalseColoring::CMFGRAD:
		cmd->config.algo = gerbil::COLOR_XYZ;
		break;
	case FalseColoring::PCA:
	case FalseColoring::PCAGRAD:
		cmd->config.algo = gerbil::COLOR_PCA;
		break;
#ifdef WITH_EDGE_DETECT
	case FalseColoring::SOM:
	case FalseColoring::SOMGRAD:
		// default parameters for false coloring (different to regular defaults)
		cmd->config.algo = gerbil::COLOR_SOM;
		cmd->config.som.maxIter = 50000;
		cmd->config.som.seed = time(NULL);

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

		break;
#endif /* WITH_EDGE_DETECT */
	default:
		assert(false);
	}
	runner->cmd = cmd;
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
	if(runner) {
		runner->terminate();
	}
}

void FalseColorModelPayload::processRunnerProgress(int percent)
{
	if(canceled) {
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
	result.convertFromImage(vole::Mat2QImage((cv::Mat3b)mat));
	emit finished(coloringType, true); // success
}

void FalseColorModelPayload::processRunnerFailure()
{
	runner->deleteLater();
	emit finished(coloringType, false); // failure
}
