#ifndef GRAPH_SEGMENTATION_MODEL_H
#define GRAPH_SEGMENTATION_MODEL_H

#include "representation.h"

#include <shared_data.h>
#include <background_task/background_task_queue.h>
#include <opencv2/core/core.hpp>

#include <QObject>
#include <QPixmap>

namespace vole
{
	class GraphSegConfig;
}

class GraphSegmentationModel : public QObject
{
	Q_OBJECT

public:
	GraphSegmentationModel(BackgroundTaskQueue *queue);
	~GraphSegmentationModel();

	// always set required iamges before using the class
	void setMultiImage(representation::t type, SharedMultiImgPtr image);

protected:
	void startGraphseg(SharedMultiImgPtr input, cv::Mat1s seedMap,
					   const vole::GraphSegConfig &config, bool resetLabel);

public slots:
	void setCurLabel(int curLabel);
	void runGraphseg(representation::t type, cv::Mat1s seedMap,
					 const vole::GraphSegConfig &config, bool resetLabel);
	void runGraphsegBand(representation::t type, int bandId, cv::Mat1s seedMap,
						 const vole::GraphSegConfig &config, bool resetLabel);

protected slots:
	void finishGraphSeg(bool success);

signals:
	/* effect: gerbil GUI enabled/disabled. */
	void setGUIEnabledRequested(bool enable, TaskType tt);

	void alterLabelRequested(short index, const cv::Mat1b &mask, bool negative);
	void seedingDone();

protected:
	typedef QMap<representation::t, SharedMultiImgPtr> ImageMap;

	BackgroundTaskQueue *const queue;
	ImageMap map;

	int curLabel;

	boost::shared_ptr<cv::Mat1s> graphsegResult;
};

#endif // GRAPH_SEGMENTATION_MODEL_H
