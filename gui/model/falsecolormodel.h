#ifndef FALSECOLOR_MODEL_H
#define FALSECOLOR_MODEL_H

#include <model/representation.h>
#include <shared_data.h>

#include <QPixmap>
#include <QMap>
#include <QMetaType>
#include <QObject>

#include <boost/any.hpp>

#include "falsecolor/falsecoloring.h"
#include "falsecolor/falsecoloringcacheitem.h"


class FalseColorModelPayload;

/**
 * @brief The FalseColorModel class provides false color image processing to
 * the GUI.
 *
 * Generally GUI objects requiring a false color representation need to
 * subscribe for it at the Controller. See
 * Controller::subscribeFalseColor(). False color image updates are
 * provided by the signal falseColoringUpdate().
 *
 */
class FalseColorModel : public QObject
{
	Q_OBJECT

public:
	FalseColorModel(QObject *parent = nullptr);
	~FalseColorModel();

	void setMultiImg(representation::t repr, SharedMultiImgPtr img);
public slots:
	void processImageUpdate(representation::t type,
	                        SharedMultiImgPtr img,
	                        bool duplicate);

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
	/** The false coloring of type coloringType has been updated.
	 *
	 * Objects requiring a false coloring need to subscribe for it at the
	 * Controller using Controller::subscribeFalseColor(). */
	void falseColoringUpdate(FalseColoring::Type coloringType, QPixmap p);

	/** The computation was cancelled as requested.
	 *
	 * This is emitted when cancelComputation() has been called before for the
	 * respective representation. It is not emitted if the computation for a
	 * representation has been cancelled by FalseColorModel internally.
	 */
	void computationCancelled(FalseColoring::Type coloringType);

	/** Signals the progress of a representation computation. */
	void progressChanged(FalseColoring::Type coloringType, int percent);

	void coloringChanged(cv::Mat3f* result);

private slots:
	/** Payload has finished computation. */
	void processComputationFinished(FalseColoring::Type coloringType,
	                                bool success);

private:
	/** Kickoff a new computation for coloringType.
	 *
	 * If there is already a computation in progress, this function has
	 * no effect.
	 */
	void computeColoring(FalseColoring::Type coloringType);

	/** Disconnect all signals from the payload object and set the cancel
	 * flag. */
	void abandonPayload(FalseColoring::Type coloringType);

	/** Allocate and reset all cache entries. */
	void resetCache();

	typedef QMap<FalseColoring::Type, FalseColorModelPayload*>
	FalseColorModelPayloadMap;

	SharedMultiImgPtr shared_img, shared_grad;
	FalseColorModelPayloadMap payloads;

	// Cache for false color results
	typedef QMap<FalseColoring::Type, FalseColoringCacheItem> FalseColoringCache;
	FalseColoringCache cache;

	// Remember pending requests we could not fulfill because of missing data.
	QMap<FalseColoring::Type, bool> pendingRequests;

	// Remember if we got imageUpdate signal for representation. Until then
	// all requests are deferred.
	QMap<representation::t, bool> representationInit;
};

#endif // FALSECOLOR_MODEL_H

