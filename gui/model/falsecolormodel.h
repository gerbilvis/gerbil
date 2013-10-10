#ifndef FALSECOLOR_MODEL_H
#define FALSECOLOR_MODEL_H

#include "representation.h"
#include <shared_data.h>

#include <QPixmap>
#include <QMap>
#include <QMetaType>
#include <QObject>

#include <boost/any.hpp>

// FIXME
// SOM-generation is not canceled when the program is terminated

class CommandRunner;
class FalseColorModelPayload;
class BackgroundTaskQueue;

/** Encapsulated enum representing the different false coloring types. */
struct FalseColoring {
	// remember to update allList static member if this is changed!
	enum Type {
		CMF=0,
		CMFGRAD,
		PCA,
		PCAGRAD,
		SOM,
		SOMGRAD
	};
	enum {SIZE=6};
	static bool isDeterministic(Type coloringTypet) {
		return !(coloringTypet == SOM || coloringTypet == SOMGRAD);
	}
	/** Returns true if the computation false coloring coloringType is based on
	 * image represesentation type and false otherwise.	 */
	static bool isBasedOn(Type coloringType, representation::t type) {
		switch (coloringType) {
		case CMF:
		case PCA:
		case SOM:
			return type == representation::IMG;
			break;
		case CMFGRAD:
		case PCAGRAD:
		case SOMGRAD:
			return type == representation::GRAD;
			break;
		default:
			assert(false);
			break;
		}
	}

	static QList<Type> all() { return allList;	}
	static size_t size() { return SIZE; }
private:
	static QList<Type> allList;
};
Q_DECLARE_METATYPE(FalseColoring)
std::ostream &operator<<(std::ostream& os, const FalseColoring::Type& coloringType);

/** Cache item for computed false color images. */
struct FalseColoringCacheItem {
	FalseColoringCacheItem(QPixmap img) : img(img), upToDate(true) {}
	QPixmap img;
	bool upToDate;
};

class FalseColorModelPayload : public QObject
{
	Q_OBJECT
public:
	FalseColorModelPayload(FalseColoring::Type coloringType,
						   SharedMultiImgPtr img,
						   SharedMultiImgPtr grad
						   )
		: canceled(false),
		  coloringType(coloringType),
		  img(img), grad(grad)
	{}

	/** Start calculation.
	 *
	 * Start new thread or whatever is necessary.
	 */
	void run();

	/** Cancel computation: Actually signal the running thread. */
	void cancel();

	QPixmap getResult() { return result; }

signals:
	/** Computation progress changed. */
	void progressChanged(FalseColoring::Type coloringType, int percent);

	/** Computation completed. */
	void finished(FalseColoring::Type coloringType, bool success = true);
private slots:
	void processRunnerSuccess(std::map<std::string, boost::any> output);
	void processRunnerFailure();
	void processRunnerProgress(int percent);
private:
	bool canceled;
	FalseColoring::Type coloringType;
	SharedMultiImgPtr img;
	SharedMultiImgPtr grad;
	CommandRunner *runner;
	QPixmap result;
};

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

