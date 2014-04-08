#include "graphsegmentationmodel.h"

#include <graphseg_config.h>
#include "graphseg_task.h"

#include <background_task/background_task_queue.h>
#include <opencv2/core/core.hpp>
#include <qtopencv.h>


GraphSegmentationModel::GraphSegmentationModel(BackgroundTaskQueue *queue)
	: queue(queue), graphsegResult(new cv::Mat1s()), curLabel(1) {}

GraphSegmentationModel::~GraphSegmentationModel() { }

void GraphSegmentationModel::setMultiImage(representation::t type,
										   SharedMultiImgPtr image)
{
	if (type == representation::IMG || type == representation::GRAD)
		map.insert(type, image);
	else // tryed setting an image of an unsupported representation type
		assert(false);
}

void GraphSegmentationModel::setCurLabel(int curLabel)
{
	this->curLabel = curLabel;
}

void GraphSegmentationModel::runGraphseg(representation::t type,
							   cv::Mat1s seedMap,
							   const vole::GraphSegConfig &config,
							   bool resetLabel)
{
	SharedMultiImgPtr input = map.value(type);
	if (!input) // image of type type was not set with setMultiImg
		assert(false);

	startGraphseg(input, seedMap, config, resetLabel);
}

void GraphSegmentationModel::runGraphsegBand(representation::t type, int bandId,
											 cv::Mat1s seedMap,
											 const vole::GraphSegConfig &config,
											 bool resetLabel)
{
	SharedMultiImgPtr img = map.value(type);
	multi_img::Band band = (**img)[bandId];
	SharedMultiImgPtr input(new SharedMultiImgBase(new multi_img(band)));
	startGraphseg(input, seedMap, config, resetLabel);
}

void GraphSegmentationModel::startGraphseg(SharedMultiImgPtr input,
										   cv::Mat1s seedMap,
										   const vole::GraphSegConfig &config,
										   bool resetLabel)
{
	emit setGUIEnabledRequested(false, TT_NONE);

	// clear current label
	if (resetLabel) {
		cv::Mat1b emptyMat;
		emit alterLabelRequested(curLabel,
								 emptyMat,
								 false);
	}

	// TODO: should this be a commandrunner instead? arguable..
	BackgroundTaskPtr taskGraphseg(new GraphSegTask(
		config, input, seedMap, graphsegResult));
	QObject::connect(taskGraphseg.get(), SIGNAL(finished(bool)),
		this, SLOT(finishGraphSeg(bool)), Qt::QueuedConnection);
	queue->push(taskGraphseg);
}

void GraphSegmentationModel::finishGraphSeg(bool success)
{
	if (success) {
		// add segmentation to current labeling
		emit alterLabelRequested(curLabel,
								 *(graphsegResult.get()),
								 false);
		// leave seeding mode for convenience
		emit seedingDone();

		emit setGUIEnabledRequested(true, TT_NONE);
	}
}
