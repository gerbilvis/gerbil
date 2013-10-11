#include "gradientcuda.h"

#include <vector>

#include <opencv2/gpu/gpu.hpp>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <stopwatch.h>

#include "multi_img/multi_img_tbb.h"
#include "rectangles.h"

#define STOPWATCH_PRINT(stopwatch, message)


bool GradientCuda::run()
{
	cv::Rect copyGlob = (*source)->roi & (*current)->roi;
	cv::Rect copySrc(0, 0, 0, 0);
	cv::Rect copyCur(0, 0, 0, 0);
	if (copyGlob.width > 0 && copyGlob.height > 0) {
		copySrc.x = copyGlob.x - (*source)->roi.x;
		copySrc.y = copyGlob.y - (*source)->roi.y;
		copySrc.width = copyGlob.width;
		copySrc.height = copyGlob.height;

		copyCur.x = copyGlob.x - (*current)->roi.x;
		copyCur.y = copyGlob.y - (*current)->roi.y;
		copyCur.width = copyGlob.width;
		copyCur.height = copyGlob.height;
	}

	std::vector<cv::Rect> calc;
	rectComplement((*source)->width, (*source)->height, copySrc, calc);

	multi_img *target = new multi_img(
		(*source)->height, (*source)->width, (*source)->size() - 1);
	if (copyGlob.width > 0 && copyGlob.height > 0) {
		for (size_t i = 0; i < target->size(); ++i) {
			multi_img::Band curBand = (*current)->bands[i](copyCur);
			multi_img::Band tgtBand = target->bands[i](copySrc);
			curBand.copyTo(tgtBand);

			if (stopper.is_group_execution_cancelled())
				break;
		}
	}

	vole::Stopwatch s;

	for (std::vector<cv::Rect>::iterator it = calc.begin(); it != calc.end(); ++it) {
		if (it->width > 0 && it->height > 0) {
			multi_img srcScope(**source, *it);
			multi_img tgtScope(*target, *it);
			std::vector<cv::gpu::GpuMat> tmpBands;
			tmpBands.reserve((*source)->size());
			for (size_t i = 0; i != (*source)->size(); ++i) {
				tmpBands.push_back(cv::gpu::GpuMat(srcScope.bands[i]));
				cv::gpu::log(tmpBands[i], tmpBands[i]);
				cv::gpu::max(tmpBands[i], 0., tmpBands[i]);
			}
			cv::gpu::GpuMat result(tgtScope.height, tgtScope.width, multi_img::ValueType);
			for (size_t i = 0; i != target->size(); ++i) {
				cv::gpu::subtract(tmpBands[i + 1], tmpBands[i], result);
				result.download(tgtScope.bands[i]);
				if (!srcScope.meta[i].empty && !srcScope.meta[i + 1].empty) {
					tgtScope.meta[i] = multi_img::BandDesc(
						srcScope.meta[i].center, srcScope.meta[i + 1].center);
				}
			}
		}

		if (stopper.is_group_execution_cancelled())
			break;
	}
	target->minval = -log((*source)->maxval);
	target->maxval = log((*source)->maxval);
	target->roi = (*source)->roi;

	// init multi_img::meta
	for (unsigned int i = 0; i < (*source)->size()-1; ++i) {
		if (!(*source)->meta[i].empty && !(*source)->meta[i+1].empty) {
			target->meta[i] = multi_img::BandDesc(
						(*source)->meta[i].center,
						(*source)->meta[i+1].center);
		}
	}

	STOPWATCH_PRINT(s, "Gradient CUDA")

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
		SharedDataSwapLock lock(current->mutex);
		current->replace(target);
		return true;
	}
}
