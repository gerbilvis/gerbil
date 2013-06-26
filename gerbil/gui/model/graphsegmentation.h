#ifndef GRAPHSEGMENTATIONMODEL_H
#define GRAPHSEGMENTATIONMODEL_H

#include "representation.h"

#include <shared_data.h>
#include <background_task_queue.h>

#include <QObject>

namespace vole
{
	class GraphSegConfig;
}

// todo: put representations in a map instead - causes problems when algo is applied on single bands

class GraphSegmentationModel : public QObject
{
	Q_OBJECT

public:
	GraphSegmentationModel(QObject *parent, BackgroundTaskQueue *queue);
	//~GraphSegmentationModel();

	// always set img and grad before using the class
	void setMultiImage(representation::t type, SharedMultiImgPtr image);

public slots:
	void runGraphseg(representation::t type,
					 const vole::GraphSegConfig &config);

protected slots:
	void finishGraphSeg(bool success);

protected:
	BackgroundTaskQueue *const queue;
	// multi image of current ROI
	SharedMultiImgPtr img, grad;
	boost::shared_ptr<cv::Mat1s> graphsegResult;
};

#endif // GRAPHSEGMENTATIONMODEL_H
