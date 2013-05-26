#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "model/image.h"
#include "model/labeling.h"
#include "mainwindow.h"
#include <background_task.h>
#include <background_task_queue.h>

#include <QObject>
#include <tbb/compat/thread>

class Controller : public QObject
{
	Q_OBJECT
public:
	explicit Controller(QString filename, bool limited_mode);
	~Controller();
	
signals:
	
public slots:
	/** requests (from GUI) */
	// change ROI, effectively spawning new image data
	void spawnROI(const cv::Rect &roi);

	// need additional label color
	void addLabel();

	/** notifies (from models, tasks) */
	// epilog task finished
	void finishTask(bool success);

	// label colors added/changed
	void propagateLabelingChange(const QVector<QColor> &colors, bool changed);

	/** internal management (maybe make protected) */
	/* this function enqueues an empty task that will signal when all previous
	 * tasks are finished. the signal will trigger enableGUINow, and the
	 * GUI gets re-enabled at the right time.
	 * The roi argument is needed to specify if the other enqueued tasks are
	 * roi-bound. If somebody cancels all tasks with that roi, our task should
	 * be cancelled as-well, and re-enable not take place.
	 */
	void enableGUILater(bool withROI = false);
	// the other side of enableGUILater
	void enableGUINow(bool forreal);
	void disableGUI(TaskType tt = TT_NONE);

private:
	// connect models with gui
	void initImage();
	void initLabeling();

	// create background thread that processes BackgroundTaskQueue
	void startQueue();
	// stop and delete thread
	// (we did not test consecutive start/stop of the queue)
	void stopQueue();

	// image model stores all multispectral image representations (IMG, GRAD, …)
	ImageModel im;

	// labeling model stores pixel/label associations and label color codes
	LabelingModel lm;

	// main window (or gui slave)
	MainWindow *window;

	BackgroundTaskQueue queue;
	std::thread *queuethread;
};

#endif // CONTROLLER_H
