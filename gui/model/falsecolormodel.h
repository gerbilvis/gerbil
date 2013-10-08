#ifndef FALSECOLOR_MODEL_H
#define FALSECOLOR_MODEL_H

#include "representation.h"
#include <shared_data.h>

#include <QPixmap>
#include <QMap>
#include <QMetaType>
#include <QObject>

#include <boost/any.hpp>

// TODO: if gradient image is changed, we do not need to delete img false color
// results and vice versa

// SOM-generation is not canceled when the program is terminated

// QImages have implicit data sharing, so the returned objects act as a
// pointer, the data is not copied If the QImage is changed in the Model, this
// change is calculated on a copy, which is then redistributed by a
// loadComplete signal

// Each request sends a loadComplete signal to ALL connected slots.  use
// QImage.cacheKey() to see if the image was changed

class CommandRunner;
class FalseColorModelPayload;
class BackgroundTaskQueue;

enum coloring {
	CMF = 0,
	PCA = 1,
	SOM = 2,
	COLSIZE = 3
};
Q_DECLARE_METATYPE(::coloring)

// struct to identify entries in the map
struct coloringWithGrad {
	coloring col;
	bool grad;

	// map needs this operator
	bool operator<(const coloringWithGrad &other) const {
		return col < other.col || (col == other.col && grad < other.grad);
	}
};

class FalseColorModel : public QObject
{
	Q_OBJECT

public:
	typedef ::coloring coloring;
	typedef FalseColorModelPayload payload;

	/* construct model without image data. Make sure to call setMultiImg()
	 * before doing any other operations with this object.
	 */
	FalseColorModel(BackgroundTaskQueue *queue);
	~FalseColorModel();

	// calls reset()
	void setMultiImg(representation::t repr, SharedMultiImgPtr img);

	// resets current true / false color representations on the next request,
	// the color images are recalculated with possibly new multi_img data
	// (CommandRunners are stopped by terminateTasksDeleteRunners())
	void reset();

public slots:
	void processImageUpdate(representation::t type, SharedMultiImgPtr img);

	// Img could be calculated in background, if the background task was
	// started before!
	void computeForeground(coloring type, bool gradient,
						   bool forceRecalculate = false);
	void computeBackground(coloring type, bool gradient,
						   bool forceRecalculate = false);
	// If image is already computed, send it back. Otherwise just leave it.
	void returnIfCached(coloring type, bool gradient);

signals:
	// Possibly check Image.cacheKey() to determine if the update is really neccessary
	void calculationComplete(coloring type, bool gradient, QPixmap img);
	void terminateRunners();

private:
	typedef QList<payload*> PayloadList;
	typedef QMap<coloringWithGrad, payload*> PayloadMap;

	// creates the runner that is neccessary for calculating the false color
	// representation of type runners are deleted in
	// terminatedTasksDeleteRunners (and therefore in reset() & ~FalseColor())
	void createRunner(coloringWithGrad mapId);

	// terminates all (queue and commandrunner) tasks and waits until the
	// terminate is complete
	void cancel();

	// terminates and resets a specific commandrunner and the specific cache
	// data this does NOT terminate queue tasks, but CMF calculation is quite
	// fast anyways
	void reset(payload *p);

	SharedMultiImgPtr shared_img, shared_grad;
	PayloadMap map;
	BackgroundTaskQueue *const queue;
};

class FalseColorModelPayload : public QObject {
	Q_OBJECT

public:
	FalseColorModelPayload(representation::t repr, FalseColorModel::coloring type, bool gradient)
		: repr(repr), type(type), gradient(gradient), runner(0)
	{}

	virtual ~FalseColorModelPayload() {}

	// signals cannot be public
	void terminateRunner() { emit requestRunnerTerminate(); }

	representation::t repr;
	FalseColorModel::coloring type;
	bool gradient;
	CommandRunner *runner;
	QPixmap img;
	qimage_ptr calcImg;  // the background task swaps its result in this
						 // variable, in taskComplete, 
						 // it is copied to img & cleared
	bool calcInProgress; // if 2 widgets request the same coloring type 
						 // before the calculation finished

public slots:
	// connect the corresponding task's finishing with this slot
	void propagateFinishedQueueTask(bool success);
	void propagateRunnerSuccess(std::map<std::string, boost::any> output);

signals:
	void requestRunnerTerminate();
	void calculationComplete(coloring type, bool gradient, QPixmap img);
};
#endif // FALSECOLOR_MODEL_H
