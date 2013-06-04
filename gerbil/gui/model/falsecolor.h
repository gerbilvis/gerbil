#ifndef MODEL_FALSECOLOR_H
#define MODEL_FALSECOLOR_H

// for representations
#include "model/image.h"
#include <background_task_queue.h>
#include <commandrunner.h>
#include <rgb.h>
#include <multi_img.h>
#include <shared_data.h>

#include <QImage>
#include <QMap>
#include <QObject>

// QImages have implicit data sharing, so the returned objects act as a pointer, the data is not copied
// If the QImage is changed in the Model, this change is calculated on a copy,
// which is then redistributed by a loadComplete signal

// Each request sends a loadComplete signal to ALL connected slots.
// use QImage.cacheKey() to see if the image was changed

class FalseColorModel : public QObject
{
	Q_OBJECT

	enum coloring {
		CMF = 0,
		PCA = 1,
		SOM = 2,
		COLSIZE = 3
	};

	struct payload {
		payload() : runner(0) {}

		CommandRunner *runner;
		QImage img;
		qimage_ptr calcImg;  // the background task swaps its result in this variable, in taskComplete, it is copied to img & cleared
		bool calcInProgress; // (if 2 widgets request the same coloring type before the calculation finished)
	};

	typedef QList<payload*> PayloadList;
	typedef QMap<coloring, payload*> PayloadMap;

public:
	/* construct model without image data. Make sure to call setMultiImg()
	 * before doing any other operations with this object.
	 */
	FalseColorModel(BackgroundTaskQueue *queue);
	~FalseColorModel();

	// calls reset()
	void setMultiImg(representation::t type, SharedMultiImgPtr img);

	// resets current true / false color representations
	// on the next request, the color images are recalculated with possibly new multi_img data
	// (CommandRunners are stopped by terminateTasksDeleteRunners())
	void reset();

public slots:
	void calculateForeground(coloring type);
	void calculateBackground(coloring type);

private slots:
	void handleFinishedQueueTask(bool finished);
	void handleRunnerSuccess(std::map<std::string, boost::any> output);

signals:
	// Possibly check Image.cacheKey() to determine if the update is really neccessary
	void calculationComplete(QImage img, coloring type);
	void terminateRunners();

private:
	// creates the runner that is neccessary for calculating the false color representation of type
	// runners are deleted in terminatedTasksDeleteRunners (and therefore in reset() & ~FalseColor())
	void createRunner(coloring type);

	// terminates all (queue and commandrunner) tasks and waits until the terminate is complete
	void cancel();

	SharedMultiImgPtr shared_img;
	PayloadMap map;
	BackgroundTaskQueue *queue;
};

#endif // MODEL_FALSECOLOR_H
