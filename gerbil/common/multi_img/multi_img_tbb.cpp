#include <cstddef>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <cv.h>

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

#include <multi_img.h>

#include "multi_img_tbb.h"

void RebuildPixels::operator()(const tbb::blocked_range<size_t> &r) const
{
	for (size_t d = r.begin(); d != r.end(); ++d) {
		multi_img::Band &src = multi.bands[d];
		multi_img::Band::const_iterator it; size_t i;
		for (it = src.begin(), i = 0; it != src.end(); ++it, ++i)
			multi.pixels[i][d] = *it;
	}
}

void ApplyCache::operator()(const tbb::blocked_range<size_t> &r) const
{
	for (size_t d = r.begin(); d != r.end(); ++d) {
		multi_img::Band &dst = multi.bands[d];
		multi_img::Band::iterator it; size_t i;
		for (it = dst.begin(), i = 0; it != dst.end(); ++it, ++i)
			*it = multi.pixels[i][d];
	}
}

void DetermineRange::operator()(const tbb::blocked_range<size_t> &r)
{
	double tmp1, tmp2;
	for (size_t d = r.begin(); d != r.end(); ++d) {
		cv::minMaxLoc(multi.bands[d], &tmp1, &tmp2);
		min = std::min<multi_img::Value>(min, (multi_img::Value)tmp1);
		max = std::max<multi_img::Value>(max, (multi_img::Value)tmp2);
	}
}

void DetermineRange::join(DetermineRange &toJoin)
{
	if (toJoin.min < min)
		min = toJoin.min;
	if (toJoin.max > max)
		max = toJoin.max;
}
