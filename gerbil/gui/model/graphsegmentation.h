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
	~GraphSegmentationModel();

	// always set img and grad before using the class
	void setMultiImage(representation::t type, SharedMultiImgPtr image);
	void setSeedMap(cv::Mat1s *seedMap);
	void setCurLabel(short *curLabelPtr);

public slots:
	void runGraphseg(representation::t type,
					 const vole::GraphSegConfig &config);

protected slots:
	void finishGraphSeg(bool success);

signals:
	void alterLabelRequested(short index, const cv::Mat1b &mask, bool negative);
	void seedingDone(bool yeah = false);

protected:
	BackgroundTaskQueue *const queue;
	cv::Mat1s *seedMap;
	short *curLabel; // That's not nice...
	// multi image of current ROI
	SharedMultiImgPtr img, grad;
	boost::shared_ptr<cv::Mat1s> graphsegResult;
};

#endif // GRAPHSEGMENTATIONMODEL_H
