#include "clampcuda.h"

#include <vector>

#include <opencv2/gpu/gpu.hpp>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <stopwatch.h>

#include "multi_img/multi_img_tbb.h"

#define STOPWATCH_PRINT(stopwatch, message)

bool ClampCuda::run()
{
	multi_img *source = &**image;

	assert(0 != source);
	multi_img *target = new multi_img(source->height, source->width, source->size());
	target->roi = source->roi;
	target->meta = source->meta;
	target->minval = (*minmax)->minval;
	target->maxval = (*minmax)->maxval;

	vole::Stopwatch s;

	cv::gpu::GpuMat band(source->height, source->width, multi_img::ValueType);
	for (size_t d = 0; d != target->size(); ++d) {
		band.upload(source->bands[d]);
		cv::gpu::max(band, target->minval, band);
		cv::gpu::min(band, target->maxval, band);
		band.download(target->bands[d]);
	}

	STOPWATCH_PRINT(s, "Clamp CUDA")

	if (includecache) {
		RebuildPixels rebuildPixels(*target);
		tbb::parallel_for(tbb::blocked_range<size_t>(0, target->size()),
			rebuildPixels, tbb::auto_partitioner(), stopper);
		target->dirty.setTo(0);
		target->anydirt = false;
	} else {
		target->resetPixels();
	}

	if (stopper.is_group_execution_cancelled()) {
		delete target;
		return false;
	} else {
		SharedDataSwapLock lock(image->mutex);
		image->replace(target);
		return true;
	}
}
