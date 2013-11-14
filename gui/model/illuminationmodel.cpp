#include "illuminationmodel.h"

#include <tbb/task_group.h>

#include "background_task/tasks/tbb/illuminanttbb.h"
#include "background_task/tasks/cuda/illuminantcuda.h"

#include <opencv2/gpu/gpu.hpp>

#include "gerbil_gui_debug.h"
#include "gerbil_config.h"

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
	if(success) { // task was not cancelled
		emit newIlluminantApplied(getIllumCoeff(i1));
		emit requestInvalidateROI(roi);
	}
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
	// currently active illuminant will be in i1
	i1 = i2;

	/* trigger re-calculation of dependent data */
	BackgroundTaskPtr taskEpilog(new BackgroundTask(roi));
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)),
		this, SLOT(finishTask(bool)), Qt::QueuedConnection);
	queue->push(taskEpilog);
}

void IllumModel::updateIllum1(int t)
{
	i1 = t;
	emit newIlluminantCurve(getIllumCoeff(i1));
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

QVector<multi_img::Value> IllumModel::getIllumCoeff(int t)
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
	QVector<multi_img::Value> cf;
	if (t > 0) {
		SharedMultiImgBaseGuard guard(*image);
		il.calcWeight((*image)->meta[0].center,
					  (*image)->meta[(*image)->size()-1].center);
		cf = QVector<multi_img::Value>::fromStdVector(
					(*image)->getIllumCoeff(il));
	}
	// else: cf is empty vector

	illuminants[t] = std::make_pair(il, cf);
}

void IllumModel::submitRemoveOldIllumTask()
{
	/* remove old illuminant */
	if (i1 != 0) {
		const Illuminant &il = getIlluminant(i1);
		if (HAVE_CUDA_GPU && USE_CUDA_ILLUMINANT) {
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

		if (HAVE_CUDA_GPU && USE_CUDA_ILLUMINANT) {
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
