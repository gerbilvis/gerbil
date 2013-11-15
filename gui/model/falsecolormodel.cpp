#include <QImage>
#include <QPixmap>
#include <opencv2/core/core.hpp>

#include <shared_data.h>

#include <multi_img.h>
#include <qtopencv.h>

#include "representation.h"
#include "../commandrunner.h"

#include "falsecolormodel.h"
#include "falsecolor/falsecolormodelpayload.h"

#include <gerbil_gui_debug.h>

QList<FalseColoring::Type> FalseColoring::allList = QList<FalseColoring::Type>()
	<< FalseColoring::CMF
	<< FalseColoring::PCA
	<< FalseColoring::PCAGRAD
	<< FalseColoring::SOM
	<< FalseColoring::SOMGRAD;

FalseColorModel::FalseColorModel()
{
	int type = QMetaType::type("FalseColoring");
	if (type == 0 || !QMetaType::isRegistered(type))
		qRegisterMetaType<FalseColoring::Type>("FalseColoring");

	type = QMetaType::type("std::map<std::string, boost::any>");
	if (type == 0 || !QMetaType::isRegistered(type))
		qRegisterMetaType< std::map<std::string, boost::any> >(
					"std::map<std::string, boost::any>");
	resetCache();
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

	resetCache();
}

void FalseColorModel::processImageUpdate(representation::t type,
										 SharedMultiImgPtr)
{
	//GGDBGM("representation " << type << endl);
	// make sure no computations based on old image data make it into the
	// cache
	FalseColorModelPayloadMap::iterator payloadIt;
	for (payloadIt = payloads.begin(); payloadIt != payloads.end(); payloadIt++) {
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
	for (it=cache.begin(); it != cache.end(); it++) {
		FalseColoring::Type coloringType = it.key();
		if(FalseColoring::isBasedOn(coloringType, type)) {
			it.value().invalidate();
			outOfDateList.append(coloringType);
		}
	}
	foreach(FalseColoring::Type coloringType, outOfDateList) {
		//GGDBGM("emit coloringOutOfDate " << coloringType << endl);
		emit coloringOutOfDate(coloringType);
	}
}

void FalseColorModel::requestColoring(FalseColoring::Type coloringType, bool recalc)
{
	//GGDBG_CALL();

	// check if we are already initialized and should deal with that request
	bool abort = false;
	{
		SharedDataLock  lock_img(shared_img->mutex);
		SharedDataLock  lock_grad(shared_grad->mutex);
		if (!shared_img || !shared_grad ||
			(**shared_img).empty() || (**shared_grad).empty())
			abort = true;
	}
	if(abort) {
		// DO NOT emit computationCancelled(coloringType);
		//GGDBGM("shared image data not ininitialized, aborting" << endl);
		return;
	}


	FalseColoringCache::iterator cacheIt = cache.find(coloringType);
	if(cacheIt != cache.end() && cacheIt->valid()) {
		if(recalc &&
				// recalc makes sense only for SOM, the other representations
				// are deterministic -> no need to recompute
				!FalseColoring::isDeterministic(coloringType))
		{
			//GGDBGM("have valid cached image, but re-calc requested for "
			//	   << coloringType << ", computing" << endl);
			computeColoring(coloringType);
		} else {
			//GGDBGM("have valid cached image for " << coloringType
			//	   << ", emitting coloringComputed" << endl);
			emit coloringComputed(coloringType, cacheIt->pixmap());
		}
	} else {
		//GGDBGM("invalid cache for "<< coloringType << ", computing." << endl);
		computeColoring(coloringType);
	}
}

void FalseColorModel::computeColoring(FalseColoring::Type coloringType)
{
	FalseColorModelPayloadMap::iterator payloadIt = payloads.find(coloringType);
	if (payloadIt != payloads.end()) {
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

void FalseColorModel::resetCache()
{
	foreach (FalseColoring::Type coloringType, FalseColoring::all()) {
		cache[coloringType].invalidate();
	}
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



