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
#include <tbb/parallel_reduce.h>

#ifdef WITH_QT
#include <QImage>
#include <QColor>
#endif

namespace MultiImg {

namespace CommonTbb {
	class RebuildPixels {
	public:
		RebuildPixels(multi_img &multi) : multi(multi) {}
		void operator()(const tbb::blocked_range<size_t> &r) const;
	private:
		multi_img &multi;
	};

	class ApplyCache {
	public:
		ApplyCache(multi_img &multi) : multi(multi) {}
		void operator()(const tbb::blocked_range<size_t> &r) const;
	private:
		multi_img &multi;
	};

	class DetermineRange {
	public:
		DetermineRange(multi_img &multi) 
			: multi(multi), min(multi_img::ValueMax), max(multi_img::ValueMin) {}
		DetermineRange(DetermineRange &toSplit, tbb::split) 
			: multi(toSplit.multi), min(multi_img::ValueMax), max(multi_img::ValueMin) {} 
		void operator()(const tbb::blocked_range<size_t> &r);
		void join(DetermineRange &toJoin);
		multi_img::Value GetMin() { return min; }
		multi_img::Value GetMax() { return max; }
	private:
		multi_img &multi;
		multi_img::Value min;
		multi_img::Value max;
	};
}

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
	BgrTbb(multi_img_ptr multi, mat_vec3f_ptr bgr,
		cv::Rect targetRoi = cv::Rect(0, 0, 0, 0)) 
		: BackgroundTask(targetRoi), multi(multi), bgr(bgr) {}
	virtual ~BgrTbb() {}
	virtual void run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	class Xyz {
	public:
		Xyz(multi_img &multi, cv::Mat_<cv::Vec3f> &xyz, size_t band, int cie) 
			: multi(multi), xyz(xyz), band(band), cie(cie) {}
		void operator()(const tbb::blocked_range2d<int> &r) const;
	private:
		multi_img &multi;
		cv::Mat_<cv::Vec3f> &xyz;
		size_t band;
		int cie;
	};

	class Bgr {
	public:
		Bgr(multi_img &multi, cv::Mat_<cv::Vec3f> &xyz, cv::Mat_<cv::Vec3f> &bgr, float greensum) 
			: multi(multi), xyz(xyz), bgr(bgr), greensum(greensum) {}
		void operator()(const tbb::blocked_range2d<int> &r) const;
	private:
		multi_img &multi;
		cv::Mat_<cv::Vec3f> &xyz;
		cv::Mat_<cv::Vec3f> &bgr;
		float greensum;
	};

	multi_img_ptr multi;
	mat_vec3f_ptr bgr;
};

#ifdef WITH_QT
class Band2QImageTbb : public BackgroundTask {
public:
	Band2QImageTbb(multi_img_ptr multi, qimage_ptr image, size_t band,
		cv::Rect targetRoi = cv::Rect(0, 0, 0, 0))
		: BackgroundTask(targetRoi), multi(multi), image(image), band(band) {}
	virtual ~Band2QImageTbb() {}
	virtual void run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	class Conversion {
	public:
		Conversion(multi_img::Band &band, QImage &image,
			multi_img::Value minval, multi_img::Value maxval)
			: band(band), image(image), minval(minval), maxval(maxval) {}
		void operator()(const tbb::blocked_range2d<int> &r) const;
	private:
		multi_img::Band &band;
		QImage &image;
		multi_img::Value minval;
		multi_img::Value maxval;
	};

	multi_img_ptr multi;
	qimage_ptr image;
	size_t band;
};
#endif

class RescaleTbb : public BackgroundTask {
public:
	RescaleTbb(multi_img_ptr source, multi_img_ptr current, size_t newsize,
		cv::Rect targetRoi = cv::Rect(0, 0, 0, 0), bool includecache = true) 
		: BackgroundTask(targetRoi), source(source), current(current), 
		newsize(newsize), includecache(includecache) {}
	virtual ~RescaleTbb() {}
	virtual void run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	class Resize {
	public:
		Resize(multi_img &source, multi_img &target, size_t newsize) 
			: source(source), target(target), newsize(newsize) {}
		void operator()(const tbb::blocked_range2d<int> &r) const;
	private:
		multi_img &source;
		multi_img &target;
		size_t newsize;
	};

	multi_img_ptr source;
	multi_img_ptr current;
	size_t newsize;
	bool includecache;
};

class GradientTbb : public BackgroundTask {
public:
	GradientTbb(multi_img_ptr source, multi_img_ptr current, 
		cv::Rect targetRoi = cv::Rect(0, 0, 0, 0), bool includecache = true) 
		: BackgroundTask(targetRoi), source(source), 
		current(current), includecache(includecache) {}
	virtual ~GradientTbb() {}
	virtual void run();
	virtual void cancel() { stopper.cancel_group_execution(); }
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
	bool includecache;
};

class PcaTbb : public BackgroundTask {
public:
	PcaTbb(multi_img_ptr source, multi_img_ptr current, unsigned int components = 0, 
		cv::Rect targetRoi = cv::Rect(0, 0, 0, 0), bool includecache = true) 
		: BackgroundTask(targetRoi), source(source), current(current), 
		components(components), includecache(includecache) {}
	virtual ~PcaTbb() {}
	virtual void run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	class Pixels {
	public:
		Pixels(multi_img &source, cv::Mat_<multi_img::Value> &target) 
			: source(source), target(target) {}
		void operator()(const tbb::blocked_range<size_t> &r) const;
	private:
		multi_img &source;
		cv::Mat_<multi_img::Value> &target;
	};

	class Projection {
	public:
		Projection(cv::Mat_<multi_img::Value> &source, multi_img &target, cv::PCA &pca) 
			: source(source), target(target), pca(pca) {}
		void operator()(const tbb::blocked_range<size_t> &r) const;
	private:
		cv::Mat_<multi_img::Value> &source;
		multi_img &target;
		cv::PCA &pca;
	};

	multi_img_ptr source;
	multi_img_ptr current;
	unsigned int components;
	bool includecache;
};

class DataRangeTbb : public BackgroundTask {
public:
	DataRangeTbb(multi_img_ptr multi, data_range_ptr range, 
		cv::Rect targetRoi = cv::Rect(0, 0, 0, 0)) 
		: BackgroundTask(targetRoi), multi(multi), range(range) {}
	virtual ~DataRangeTbb() {}
	virtual void run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	multi_img_ptr multi;
	data_range_ptr range;
};

}

#endif
