/*	
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef MULTI_IMG_TASKS_H
#define MULTI_IMG_TASKS_H

#include "background_task.h"
#include "shared_data.h"
#include <multi_img.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <tbb/task.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/partitioner.h>
#include <tbb/parallel_for.h>

namespace MultiImg {

class BgrSerial : public BackgroundTask {
public:
	BgrSerial(multi_img_ptr multi, mat_vec3f_ptr bgr) 
		: multi(multi), bgr(bgr) {}
	virtual ~BgrSerial() {};
	virtual void run();
	virtual void cancel() {}
protected:
	multi_img_ptr multi;
	mat_vec3f_ptr bgr;
};

class BgrTbb : public BackgroundTask {
public:
	BgrTbb(multi_img_ptr multi, mat_vec3f_ptr bgr) 
		: multi(multi), bgr(bgr) {}
	virtual ~BgrTbb() {}
	void run();
	void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	class Xyz {
	public:
		Xyz(multi_img_ptr &multi, cv::Mat_<cv::Vec3f> &xyz, size_t band, int cie) 
			: multi(multi), xyz(xyz), band(band), cie(cie) {}
		void operator()(const tbb::blocked_range2d<int> &r) const;
	private:
		multi_img_ptr &multi;
		cv::Mat_<cv::Vec3f> &xyz;
		size_t band;
		int cie;
	};

	class Bgr {
	public:
		Bgr(multi_img_ptr &multi, cv::Mat_<cv::Vec3f> &xyz, cv::Mat_<cv::Vec3f> &bgr, float greensum) 
			: multi(multi), xyz(xyz), bgr(bgr), greensum(greensum) {}
		void operator()(const tbb::blocked_range2d<int> &r) const;
	private:
		multi_img_ptr &multi;
		cv::Mat_<cv::Vec3f> &xyz;
		cv::Mat_<cv::Vec3f> &bgr;
		float greensum;
	};

	multi_img_ptr multi;
	mat_vec3f_ptr bgr;
};

class GradientTbb : public BackgroundTask {
public:
	GradientTbb(multi_img_ptr source, multi_img_ptr current) 
		: source(source), current(current) {}
	virtual ~GradientTbb() {}
	void run();
	void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	class Log {
	public:
		Log(multi_img &source, multi_img &target) 
			: source(source), target(target) {}
		void operator()(const tbb::blocked_range<size_t> &r) const;
	private:
		multi_img &source;
		multi_img &target;
	};

	class Grad {
	public:
		Grad(multi_img &source, multi_img &target) 
			: source(source), target(target) {}
		void operator()(const tbb::blocked_range<size_t> &r) const;
	private:
		multi_img &source;
		multi_img &target;
	};

	multi_img_ptr source;
	multi_img_ptr current;
};

}

#endif
