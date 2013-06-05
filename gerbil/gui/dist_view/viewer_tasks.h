#ifndef MULTI_IMG_VIEWER_TASKS_H
#define MULTI_IMG_VIEWER_TASKS_H

#include "compute.h"

#include <multi_img.h>

#include <tbb/task.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/partitioner.h>
#include <tbb/parallel_for.h>

struct fillMaskSingleBody {
	cv::Mat1b &mask;
	const multi_img::Band &band;
	int dim;
	int sel;
	multi_img::Value minval;
	multi_img::Value binsize;
	const std::vector<multi_img::Value> &illuminant;

	fillMaskSingleBody(cv::Mat1b &mask, const multi_img::Band &band, int
			dim, int sel, multi_img::Value minval, multi_img::Value binsize,
			const std::vector<multi_img::Value> &illuminant);

	void operator()(const tbb::blocked_range2d<size_t> &r) const;
};

struct fillMaskLimitersBody {
	cv::Mat1b &mask;
	const multi_img &image;
	multi_img::Value minval;
	multi_img::Value binsize;
	const std::vector<multi_img::Value> &illuminant;
	const std::vector<std::pair<int, int> > &l;

	fillMaskLimitersBody(cv::Mat1b &mask, const multi_img &image,
		multi_img::Value minval, multi_img::Value binsize,
		const std::vector<multi_img::Value> &illuminant,
		const std::vector<std::pair<int, int> > &l);

	void operator()(const tbb::blocked_range2d<size_t> &r) const;
};

struct updateMaskLimitersBody {
	cv::Mat1b &mask;
	const multi_img &image;
	int dim;
	multi_img::Value minval;
	multi_img::Value binsize;
	const std::vector<multi_img::Value> &illuminant;
	const std::vector<std::pair<int, int> > &l;

	updateMaskLimitersBody(cv::Mat1b &mask, const multi_img &image, int dim,
		multi_img::Value minval, multi_img::Value binsize,
		const std::vector<multi_img::Value> &illuminant,
		const std::vector<std::pair<int, int> > &l);


	void operator()(const tbb::blocked_range2d<size_t> &r) const;
};

#endif // MULTI_IMG_VIEWER_TASKS_H
