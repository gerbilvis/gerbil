#ifndef FALSECOLOR_H
#define FALSECOLOR_H

#include <background_task_queue.h>
#include <commandrunner.h>
#include <rgb.h>
#include <multi_img.h>
#include <shared_data.h>

#include <QImage>
#include <QMap>
#include <QObject>

class FalseColor : public QObject
{
	Q_OBJECT

	enum coloring {
		CMF = 0,
		PCA = 1,
		SOM = 2,
		COLSIZE = 3
	};

	struct payload {
		CommandRunner *runner;
		gerbil::RGB *cmd;	 // just a shortcut, avoids casting and long code
		QImage img;
		qimage_ptr calcImg;  // the background task swaps its result in this variable, in taskComplete, it is copied to img & cleared
		bool calcInProgress; // (if 2 widgets request the same coloring type before the calculation finished)
	};

	typedef QList<payload*> PayloadList;
	typedef QMap<coloring, payload*> PayloadMap;

public:
	FalseColor(SharedMultiImgPtr shared_img, const BackgroundTaskQueue queue);
	//FalseColor(const multi_img& img, const BackgroundTaskQueue queue);
	~FalseColor();

	// resets current true / false color representations
	// on the next request, the color images are recalculated with possibly new multi_img data
	void resetCaches();

	// always calls resetCaches()
	void setMultiImg(SharedMultiImgPtr img);
	//void setMultiImg(const multi_img& img);

public slots:
	void requestForeground(coloring type);
	void requestBackground(coloring type);

private slots:
	void queueTaskFinished(bool finished);
	void runnerSuccess(std::map<std::string, boost::any> output);

signals:
	void loadComplete(QImage img, coloring type, bool changed);

private:
	SharedMultiImgPtr shared_img; // not const - wird langfristig zu einem pointer
	//const multi_img *img; // geht evtl nicht, weil mans beim task starten uebergeben muss - langfristig schoener?
	PayloadMap map;
	BackgroundTaskQueue &queue;
};

#endif // FALSECOLOR_H
