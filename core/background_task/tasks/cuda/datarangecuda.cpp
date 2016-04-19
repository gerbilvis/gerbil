#include "datarangecuda.h"

#ifdef GERBIL_CUDA

#include <vector>
#include <algorithm>

#include <opencv2/gpu/gpu.hpp>
#include <tbb/blocked_range2d.h>

#include <stopwatch.h>

#define STOPWATCH_PRINT(stopwatch, message)

bool DataRangeCuda::run()
{
	Stopwatch s;

	double tmp1, tmp2;
	multi_img::Value min = multi_img::ValueMax;
	multi_img::Value max = multi_img::ValueMin;
	cv::gpu::GpuMat band((*multi)->height, (*multi)->width, multi_img::ValueType);
	for (size_t d = 0; d != (*multi)->size(); ++d) {
		band.upload((*multi)->bands[d]);
		cv::gpu::minMaxLoc(band, &tmp1, &tmp2);
		min = std::min<multi_img::Value>(min, (multi_img::Value)tmp1);
		max = std::max<multi_img::Value>(max, (multi_img::Value)tmp2);
	}

	STOPWATCH_PRINT(s, "DataRange CUDA")

	if (!stopper.is_group_execution_cancelled()) {
		SharedDataSwapLock lock(range->mutex);
		(*range)->min = min;
		(*range)->max = max;
		return true;
	} else {
		return false;
	}
}

#endif
