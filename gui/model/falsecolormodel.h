#ifndef FALSECOLOR_MODEL_H
#define FALSECOLOR_MODEL_H

#include "representation.h"
#include <shared_data.h>

#include <QPixmap>
#include <QMap>
#include <QMetaType>
#include <QObject>

#include <boost/any.hpp>

#include "falsecolor/falsecoloring.h"
#include "falsecolor/falsecoloringcacheitem.h"

// FIXME
// SOM-generation is not canceled when the program is terminated

class FalseColorModelPayload;

/**
 * @brief The FalseColorModel class provides false color image processing to the GUI.
 *
 *  FalseColorModel will send a coloringOutOfDate() signal once it has the
 *  necessary data to compute the given FalseColoring::Type. Clients should not
 *  requestColoring() until a coloringOutOfDate() for a FalseColoring::Type has
 *  been signalled.
 *
 */
class FalseColorModel : public QObject
{
	Q_OBJECT

public:
	/* Constructor.
	 */
	FalseColorModel();
	~FalseColorModel();

	void setMultiImg(representation::t repr, SharedMultiImgPtr img);
public slots:
	void processImageUpdate(representation::t type, SharedMultiImgPtr img);

	/** Request a rendering of coloringType of the current image or
	 * gradient and ROI.
	 *
	 * If FalseColorModel is not initialized, request will be ignored.
	 *
	 * @param recalc If set, the result shall be recalculated wether or not an
	 *               up-to-date cached copy is available. Useful to request a
	 *               new SOM.
	 */
	void requestColoring(FalseColoring::Type coloringType, bool recalc = false);


	/** Cancels a previously requested calculation for coloringType. */
	void cancelComputation(FalseColoring::Type coloringType);

signals:
	/** The false coloring image is not up-to-date anymore.
	 *
	 * Updated image data was received and the last false color rendering of
	 * coloringType does not match the new image data. A new false coloring
	 * will be computed on request.
	 *
	 * @param coloringType The false coloring type that has become invalid.
	 */
	void coloringOutOfDate(FalseColoring::Type coloringType);

	void coloringComputed(FalseColoring::Type coloringType, QPixmap p);

	/** The computation was cancelled as requested.
	 *
	 *	Not used for signalling abort by FalseColorModel itself.
	 */
	void computationCancelled(FalseColoring::Type coloringType);

	void progressChanged(FalseColoring::Type coloringType, int percent);

private slots:
	/** Payload is done computing. */
	void processComputationFinished(FalseColoring::Type coloringType,
									bool success);

private:
	/** Kickoff a new computation for coloringType.
	 *
	 * If there is already a computation in progress, this function has
	 * no effect.
	 */
	void computeColoring(FalseColoring::Type coloringType);
	/** Allocate and reset all cache entries. */
    void resetCache();

	//typedef QList<payload*> PayloadList;
	typedef QMap<FalseColoring::Type, FalseColorModelPayload*>
			FalseColorModelPayloadMap;

	SharedMultiImgPtr shared_img, shared_grad;
	FalseColorModelPayloadMap payloads;

	// this is were we store results
	typedef QMap<FalseColoring::Type, FalseColoringCacheItem> FalseColoringCache;
	FalseColoringCache cache;
};

#endif // FALSECOLOR_MODEL_H

