#include "graphsegmentation.h"

#include <graphseg_config.h>
#include "../tasks/graphsegbackground.h"

#include <background_task_queue.h>


GraphSegmentationModel::GraphSegmentationModel(QObject *parent,
											   BackgroundTaskQueue *queue)
	: QObject(parent), queue(queue), graphsegResult(new cv::Mat1s())
{
}

void GraphSegmentationModel::setMultiImage(representation::t type,
										   SharedMultiImgPtr mulit_image)
{
	switch (type)
	{
	case representation::IMG:
		img = mulit_image;
	case representation::GRAD:
		grad = mulit_image;
	default: // tryed setting an image of an unsupported representation type
		assert(false);
	}
}

void GraphSegmentationModel::runGraphseg(representation::t type,
							   const vole::GraphSegConfig &config)
{
	SharedMultiImgPtr input;
	switch (type)
	{
	case representation::IMG:
		input = img;
	case representation::GRAD:
		input = grad;
	default:
		assert(false);
	}

	// TODO: why disable GUI? Where is it enabled? -> do it in finishGraphSeg
	// TODO: build signal requestGUI
	// setGUIEnabled(false);
	// TODO: should this be a commandrunner instead? arguable..
	BackgroundTaskPtr taskGraphseg(new GraphsegBackground(
		config, input, bandView->seedMap, graphsegResult));
	QObject::connect(taskGraphseg.get(), SIGNAL(finished(bool)),
		this, SLOT(finishGraphSeg(bool)), Qt::QueuedConnection);
	queue->push(taskGraphseg);
}

void GraphSegmentationModel::finishGraphSeg(bool success)
{
	/*
	 * @ploner probably doesn't work yet (?).
	if (success) {
		// add segmentation to current labeling
		emit alterLabelRequested(bandView->getCurLabel(),
								 *(graphsegResult.get()), false);
		// leave seeding mode for convenience
		emit seedingDone();
	}
	*/
}
