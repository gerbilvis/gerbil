#include "illuminationmodel.h"

#include <tbb/task_group.h>

#include "background_task/tasks/tbb/illuminanttbb.h"
#include "background_task/tasks/cuda/illuminantcuda.h"

#include <opencv2/gpu/gpu.hpp>

#include "gerbil_gui_debug.h"

#define USE_CUDA_ILLUMINANT     0

IllumModel::IllumModel(BackgroundTaskQueue *queue)
	: i1(0), i2(0), queue(queue)
{
}

void IllumModel::setMultiImage(SharedMultiImgPtr image)
{
	this->image = image;
}


void IllumModel::finishTask(bool success)
{
	if(success) // task was not cancelled
		emit requestInvalidateROI(roi);
}

// TODO: part of controller!
void IllumModel::applyIllum()
{
	queue->cancelTasks();
	// FIXME re-apply illuminant while calculation in progess is currently
	// not implemented (?) and probably broken.
	emit setGUIEnabledRequested(false, TT_APPLY_ILLUM);

	submitRemoveOldIllumTask();
	submitAddNewIllumTask();


	if (i2>0) {
		cv::Mat1f il = getIllumCoeff(i2);
		emit newIlluminant(il);
		emit illuminantIsApplied(true);
	} else	{
		// back to neutral
		emit illuminantIsApplied(false);
	}

	BackgroundTaskPtr taskEpilog(new BackgroundTask(roi));
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)),
		this, SLOT(finishTask(bool)), Qt::QueuedConnection);
	queue->push(taskEpilog);
}

void IllumModel::updateIllum1(int t)
{
	i1 = t;
	cv::Mat1f il = getIllumCoeff(t);
	emit newIlluminant(il);
}

void IllumModel::updateIllum2(int t)
{
	i2 = t;
}

void IllumModel::setRoi(cv::Rect roi)
{
	this->roi = roi;
}

const Illuminant & IllumModel::getIlluminant(int t)
{
	Illum_map::iterator i = illuminants.find(t);
	if (i != illuminants.end()) {
		return i->second.first;
	}

	buildIllum(t);
	return illuminants[t].first;
}

cv::Mat1f IllumModel::getIllumCoeff(int t)
{
	Illum_map::iterator i = illuminants.find(t);
	if (i != illuminants.end()) {
		return i->second.second;
	}

	buildIllum(t);
	return illuminants[t].second;
}

void IllumModel::buildIllum(int t)
{
	Illuminant il(t);
	cv::Mat1f cf;
	{
		SharedMultiImgBaseGuard guard(*image);
		il.calcWeight((*image)->meta[0].center,
					  (*image)->meta[(*image)->size()-1].center);
		std::vector<float> cfv = (*image)->getIllumCoeff(il);
		cf = cv::Mat1f(cfv, /* copy */ true);

	}
	illuminants[t] = std::make_pair(il, cf);
}

void IllumModel::submitRemoveOldIllumTask()
{
	/* remove old illuminant */
	if (i1 != 0) {
		const Illuminant &il = getIlluminant(i1);

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_ILLUMINANT) {
			BackgroundTaskPtr taskIllum(new IlluminantCuda(
				image, il, true, roi, false));
			queue->push(taskIllum);
		} else {
			BackgroundTaskPtr taskIllum(new IlluminantTbb(
				image, il, true, roi, false));
			queue->push(taskIllum);
		}
	}
}

void IllumModel::submitAddNewIllumTask()
{
	/* add new illuminant */
	if (i2 != 0) {
		const Illuminant &il = getIlluminant(i2);

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_ILLUMINANT) {
			BackgroundTaskPtr taskIllum(new IlluminantCuda(
				image, il, false, roi, false));
			queue->push(taskIllum);
		} else {
			BackgroundTaskPtr taskIllum(new IlluminantTbb(
				image, il, false, roi, false));
			queue->push(taskIllum);
		}
	}
}
