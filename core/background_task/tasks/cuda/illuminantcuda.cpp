#include "illuminantcuda.h"



#include <vector>

#include <opencv2/gpu/gpu.hpp>
//#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <stopwatch.h>

#include "multi_img/multi_img_tbb.h"
//#include "rectangles.h"

#define STOPWATCH_PRINT(stopwatch, message)

bool IlluminantCuda::run()
{
	multi_img *source = &**multi;
	assert(0 != source);
	multi_img *target = new multi_img(source->height, source->width, source->size());
	target->roi = source->roi;
	target->meta = source->meta;
	target->minval = source->minval;
	target->maxval = source->maxval;

	vole::Stopwatch s;

	cv::gpu::GpuMat band(source->height, source->width, multi_img::ValueType);
	if (remove) {
		for (size_t d = 0; d != target->size(); ++d) {
			band.upload(source->bands[d]);
			cv::gpu::divide(band, (multi_img::Value)il.at(source->meta[d].center), band);
			band.download(target->bands[d]);
		}
	} else {
		for (size_t d = 0; d != target->size(); ++d) {
			band.upload(source->bands[d]);
			cv::gpu::multiply(band, (multi_img::Value)il.at(source->meta[d].center), band);
			band.download(target->bands[d]);
		}
	}

	STOPWATCH_PRINT(s, "Illuminant CUDA")

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
		SharedDataSwapLock lock(multi->mutex);
		multi->replace(target);
		return true;
	}
}
