/*
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef MULTI_IMG_TASKS_H
#define MULTI_IMG_TASKS_H

#ifdef WITH_BOOST_THREAD
#include <background_task/background_task.h>
#include <shared_data.h>
#include <multi_img/illuminant.h>
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

enum NormMode {
	NORM_OBSERVED = 0,
	NORM_THEORETICAL = 1,
	NORM_FIXED = 2
};


//class BgrSerial : public BackgroundTask {
//public:
//	BgrSerial(SharedMultiImgPtr multi, mat3f_ptr bgr,
//		cv::Rect targetRoi = cv::Rect())
//		: BackgroundTask(targetRoi), multi(multi), bgr(bgr) {}
//	virtual ~BgrSerial() {}
//	virtual bool run();
//	virtual void cancel() {}
//protected:
//	SharedMultiImgPtr multi;
//	mat3f_ptr bgr;
//};

#ifdef WITH_QT

#endif


class GradientTbb : public BackgroundTask {
public:
	GradientTbb(SharedMultiImgPtr source, SharedMultiImgPtr current,
		cv::Rect targetRoi = cv::Rect(), bool includecache = true) 
		: BackgroundTask(targetRoi), source(source), 
		current(current), includecache(includecache) {}
	virtual ~GradientTbb() {}
	virtual bool run();
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

	SharedMultiImgPtr source;
	SharedMultiImgPtr current;
	bool includecache;
};

class GradientCuda : public BackgroundTask {
public:
	GradientCuda(SharedMultiImgPtr source, SharedMultiImgPtr current,
		cv::Rect targetRoi = cv::Rect(), bool includecache = true) 
		: BackgroundTask(targetRoi), source(source), 
		current(current), includecache(includecache) {}
	virtual ~GradientCuda() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;
	SharedMultiImgPtr source;
	SharedMultiImgPtr current;
	bool includecache;
};

class PcaTbb : public BackgroundTask {
public:
	PcaTbb(SharedMultiImgPtr source, SharedMultiImgPtr current, unsigned int components = 0,
		cv::Rect targetRoi = cv::Rect(), bool includecache = true) 
		: BackgroundTask(targetRoi), source(source), current(current), 
		components(components), includecache(includecache) {}
	virtual ~PcaTbb() {}
	virtual bool run();
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

	SharedMultiImgPtr source;
	SharedMultiImgPtr current;
	unsigned int components;
	bool includecache;
};

class DataRangeTbb : public BackgroundTask {
public:
	DataRangeTbb(SharedMultiImgPtr multi, SharedMultiImgRangePtr range,
		cv::Rect targetRoi = cv::Rect()) 
		: BackgroundTask(targetRoi), multi(multi), range(range) {}
	virtual ~DataRangeTbb() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	SharedMultiImgPtr multi;
	SharedMultiImgRangePtr range;
};

class DataRangeCuda : public BackgroundTask {
public:
	DataRangeCuda(SharedMultiImgPtr multi, SharedMultiImgRangePtr range,
		cv::Rect targetRoi = cv::Rect()) 
		: BackgroundTask(targetRoi), multi(multi), range(range) {}
	virtual ~DataRangeCuda() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	SharedMultiImgPtr multi;
	SharedMultiImgRangePtr range;
};

class ClampTbb : public BackgroundTask {
public:
	ClampTbb(SharedMultiImgPtr image, SharedMultiImgPtr minmax,
			 cv::Rect targetRoi = cv::Rect(), bool includecache = true)
		: BackgroundTask(targetRoi), image(image),	minmax(minmax),
		  includecache(includecache)
	{}
	virtual ~ClampTbb() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	class Clamp {
	public:
		Clamp(multi_img &source, multi_img &target) : source(source), target(target) {}
		void operator()(const tbb::blocked_range<size_t> &r) const;
	private:
		multi_img &source;
		multi_img &target;
	};

	SharedMultiImgPtr image;

	SharedMultiImgPtr minmax;
	bool includecache;
};

class ClampCuda : public BackgroundTask {
public:
	ClampCuda(SharedMultiImgPtr image, SharedMultiImgPtr minmax,
		cv::Rect targetRoi = cv::Rect(), bool includecache = true) 
		: BackgroundTask(targetRoi), image(image), minmax(minmax), includecache(includecache)
	{}
	virtual ~ClampCuda() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;
	SharedMultiImgPtr image;
	SharedMultiImgPtr minmax;
	bool includecache;
};

class IlluminantTbb : public BackgroundTask {
public:
	IlluminantTbb(SharedMultiImgPtr multi, const Illuminant& il, bool remove,
		cv::Rect targetRoi = cv::Rect(), bool includecache = true) 
		: BackgroundTask(targetRoi), multi(multi), 
		il(il), remove(remove), includecache(includecache) {}
	virtual ~IlluminantTbb() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	class Illumination {
	public:
		Illumination(multi_img &source, multi_img &target, Illuminant& il, bool remove) 
			: source(source), target(target), il(il), remove(remove) {}
		void operator()(const tbb::blocked_range<size_t> &r) const;
	private:
		multi_img &source;
		multi_img &target;
		Illuminant &il; 
		bool remove;
	};

	SharedMultiImgPtr multi;
	Illuminant il; 
	bool remove;
	bool includecache;
};

class IlluminantCuda : public BackgroundTask {
public:
	IlluminantCuda(SharedMultiImgPtr multi, const Illuminant& il, bool remove,
		cv::Rect targetRoi = cv::Rect(), bool includecache = true) 
		: BackgroundTask(targetRoi), multi(multi), 
		il(il), remove(remove), includecache(includecache) {}
	virtual ~IlluminantCuda() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	SharedMultiImgPtr multi;
	Illuminant il; 
	bool remove;
	bool includecache;
};

}

#endif
#endif
