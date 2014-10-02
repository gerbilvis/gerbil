#include <QImage>
#include <QPixmap>
#include <QSet>
#include <opencv2/core/core.hpp>

#include <shared_data.h>

#include <multi_img.h>
#include <qtopencv.h>

#include "representation.h"
#include "../commandrunner.h"

#include "falsecolormodel.h"
#include "falsecolor/falsecolormodelpayload.h"

//#define GGDBG_MODULE
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

	foreach(FalseColoring::Type c, FalseColoring::all()) {
		pendingRequests[c] = false;
	}
	foreach(representation::t type, representation::all()) {
		representationInit[type] = false;
	}

	resetCache();
}

FalseColorModel::~FalseColorModel()
{
	// nothing to do
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
										 SharedMultiImgPtr,
										 bool duplicate)
{
	GGDBGM("representation " << type << ", "
		   << "duplicate: " << duplicate << endl);
	if (duplicate) {
		return;
	}
	// make sure no computations based on old image data make it into the
	// cache
	FalseColorModelPayloadMap::iterator payloadIt;
	for (payloadIt = payloads.begin(); payloadIt != payloads.end(); payloadIt++) {
		FalseColoring::Type coloringType = payloadIt.key();
		if(FalseColoring::isBasedOn(coloringType, type)) {
			GGDBGM("abandoning payload " << coloringType << endl);
			abandonPayload(coloringType);
			// remember to restart the calculation
			pendingRequests[coloringType] = true;
		}
	}

	// invalidate affected cache entries:
	FalseColoringCache::iterator it;
	for (it=cache.begin(); it != cache.end(); it++) {
		FalseColoring::Type coloringType = it.key();
		if(FalseColoring::isBasedOn(coloringType, type)) {
			GGDBGM("invalidate cache for " << type << endl);
			it.value().invalidate();
		}
	}

	foreach(FalseColoring::Type c, FalseColoring::all()) {
		if (FalseColoring::isBasedOn(c, type) && pendingRequests[c]) {
			GGDBGM("processing pending request for " << c << endl);
			pendingRequests[c] = false;
			computeColoring(c);
		}
	}

	representationInit[type] = true;
}

void FalseColorModel::requestColoring(FalseColoring::Type coloringType, bool recalc)
{
	GGDBG_CALL();

	// check if we are already initialized and should deal with that request
	foreach(representation::t repr, representation::all()) {
		if (FalseColoring::isBasedOn(coloringType, repr) && !representationInit[repr]) {
			GGDBGM("request for " << coloringType <<
				   ": representation " << repr << " not initialized, deferring request"<< endl);
			pendingRequests[coloringType] = true;
			return;
		}
	}

	FalseColoringCache::iterator cacheIt = cache.find(coloringType);
	if(cacheIt != cache.end() && cacheIt->valid()) {
		if(recalc &&
				// recalc makes sense only for SOM, the other representations
				// are deterministic -> no need to recompute
				!FalseColoring::isDeterministic(coloringType))
		{
			GGDBGM("have valid cached image, but re-calc requested for "
				   << coloringType << ", computing" << endl);
			cache[coloringType].invalidate();
			computeColoring(coloringType);
		} else {
			GGDBGM("have valid cached image for " << coloringType
				   << ", emitting coloringComputed" << endl);
			emit coloringComputed(coloringType, cacheIt->pixmap());
		}
	} else {
		GGDBGM("invalid cache for "<< coloringType << ", computing." << endl);
		computeColoring(coloringType);
	}
}

void FalseColorModel::computeColoring(FalseColoring::Type coloringType)
{
	FalseColorModelPayloadMap::iterator payloadIt = payloads.find(coloringType);
	if (payloadIt != payloads.end()) {
		//GGDBGM("computation in progress for "<< coloringType << endl);
		// computation in progress
		return;
	}

	//GGDBGM("computation starts for "<< coloringType << endl);
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

void FalseColorModel::abandonPayload(FalseColoring::Type coloringType) {
	FalseColorModelPayload *payload = NULL;
	FalseColorModelPayloadMap::iterator payloadIt = payloads.find(coloringType);
	if(payloadIt != payloads.end()) {
		payload = *payloadIt;
		assert(payload);
		disconnect(payload, 0, this, 0);
		payload->cancel();
		payload->deleteLater();
		payloads.erase(payloadIt);
	}
}

void FalseColorModel::resetCache()
{
	foreach (FalseColoring::Type coloringType, FalseColoring::all()) {
		cache[coloringType].invalidate();
	}
}

void FalseColorModel::cancelComputation(FalseColoring::Type coloringType)
{
	GGDBGM(coloringType<<endl);
	//cache[coloringType].invalidate();
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
		//GGDBGM("emitting coloringComputed " << coloringType<<endl);
		emit coloringComputed(coloringType, pixmap);
	} else {
		//GGDBGM("emitting computationCancelled " << coloringType<<endl);
		emit computationCancelled(coloringType);
	}
}



