#include "graphsegmentation.h"

#include <graphseg_config.h>
#include "../tasks/graphsegbackground.h"

#include <background_task_queue.h>


GraphSegmentationModel::GraphSegmentationModel(QObject *parent,
											   BackgroundTaskQueue *queue)
	: QObject(parent), queue(queue), graphsegResult(new cv::Mat1s()) { }

GraphSegmentationModel::~GraphSegmentationModel() { }

void GraphSegmentationModel::setMultiImage(representation::t type,
										   SharedMultiImgPtr mulit_image)
{
	switch (type)
	{
	case representation::IMG:
		img = mulit_image;
		break;
	case representation::GRAD:
		grad = mulit_image;
		break;
	default: // tryed setting an image of an unsupported representation type
		assert(false);
	}
}

void GraphSegmentationModel::setSeedMap(cv::Mat1s *seedMap)
{
	this->seedMap = seedMap;
}

void GraphSegmentationModel::setCurLabel(short *curLabelPtr)
{
	this->curLabel = curLabelPtr;
}

void GraphSegmentationModel::runGraphseg(representation::t type,
							   const vole::GraphSegConfig &config)
{
	SharedMultiImgPtr input;
	switch (type)
	{
	case representation::IMG:
		input = img;
		break;
	case representation::GRAD:
		input = grad;
		break;
	default:
		assert(false);
	}

	// TODO: why disable GUI? Where is it enabled? -> do it in finishGraphSeg
	// TODO: build signal requestGUI
	// setGUIEnabled(false);
	// TODO: should this be a commandrunner instead? arguable..
	BackgroundTaskPtr taskGraphseg(new GraphsegBackground(
		config, input, *seedMap, graphsegResult));
	QObject::connect(taskGraphseg.get(), SIGNAL(finished(bool)),
		this, SLOT(finishGraphSeg(bool)), Qt::QueuedConnection);
	queue->push(taskGraphseg);
}

void GraphSegmentationModel::finishGraphSeg(bool success)
{
	if (success) {
		// add segmentation to current labeling
		emit alterLabelRequested(*curLabel, *(graphsegResult.get()), false);
		// leave seeding mode for convenience
		emit seedingDone();
	}
}
