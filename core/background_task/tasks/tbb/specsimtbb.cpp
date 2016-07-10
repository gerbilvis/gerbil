#include "specsimtbb.h"
#include "band2qimagetbb.h"

#include <multi_img.h>
#include <background_task/background_task.h>

#include <shared_data.h>

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

bool SpecSimTbb::run()
{
	multi_img::Band result((*multi)->height, (*multi)->width);
	const multi_img::Pixel& reference = (**multi)(coord.y,coord.x);

	tbb::parallel_for(tbb::blocked_range2d<size_t>(
						  0, (*multi)->height, 0, (*multi)->width),
					  [&](tbb::blocked_range2d<size_t> r) {
		for (size_t y = r.rows().begin(); y != r.rows().end(); ++y) {
			for (size_t x = r.cols().begin(); x != r.cols().end(); ++x) {
				// negate so small values get high response
				result(y,x) = -1.f*(float)distfun->getSimilarity((**multi)(y,x), reference);
			}
		}
	});

	double min;
	double max;
	cv::minMaxLoc(result, &min, &max);
	multi_img::Value minval = (multi_img::Value)min;
	multi_img::Value maxval = 0; // 0 -> equal // (multi_img::Value)max;

	QImage *target = new QImage(result.cols, result.rows, QImage::Format_ARGB32);
	Band2QImage computeConversion(result, *target, minval, maxval);
	tbb::parallel_for(tbb::blocked_range2d<int>(0, result.rows, 0, result.cols),
					  computeConversion, tbb::auto_partitioner(), stopper);

	if (stopper.is_group_execution_cancelled()) {
		delete target;
		return false;
	} else {
		SharedDataSwapLock lock(image->mutex);
		image->replace(target);
		return true;
	}
}
