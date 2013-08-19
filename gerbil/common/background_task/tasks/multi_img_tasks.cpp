/*	
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifdef WITH_BOOST_THREAD
#include "multi_img_tasks.h"

#include <stopwatch.h>
#include <opencv2/gpu/gpu.hpp>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>


#include <rectangles.h>
#include <multi_img/multi_img_tbb.h>
#include <multi_img/cieobserver.h>

//#define STOPWATCH_PRINT(stopwatch, message) stopwatch.print_reset(message);
#define STOPWATCH_PRINT(stopwatch, message)


namespace MultiImg {


//bool BgrSerial::run()
//{
//	cv::Mat_<cv::Vec3f> *newBgr = new cv::Mat_<cv::Vec3f>();
//	*newBgr = (*multi)->bgr();
//	SharedDataSwapLock lock(bgr->mutex);
//	delete bgr->swap(newBgr);
//	return true;
//}


#ifdef WITH_QT

#endif




bool GradientTbb::run() 
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

	multi_img temp((*source)->height, (*source)->width, (*source)->size());
	for (std::vector<cv::Rect>::iterator it = calc.begin(); it != calc.end(); ++it) {
		if (it->width > 0 && it->height > 0) {
			multi_img srcScope(**source, *it);
			multi_img tmpScope(temp, *it);
			Log computeLog(srcScope, tmpScope);
			tbb::parallel_for(tbb::blocked_range<size_t>(0, temp.size()), 
				computeLog, tbb::auto_partitioner(), stopper);
		}

		if (stopper.is_group_execution_cancelled()) 
			break;
	}
	temp.minval = 0.;
	temp.maxval = log((*source)->maxval);
	temp.roi = (*source)->roi;

	for (std::vector<cv::Rect>::iterator it = calc.begin(); it != calc.end(); ++it) {
		if (it->width > 0 && it->height > 0) {
			multi_img tmpScope(temp, *it);
			multi_img tgtScope(*target, *it);
			Grad computeGrad(tmpScope, tgtScope);
			tbb::parallel_for(tbb::blocked_range<size_t>(0, target->size()), 
				computeGrad, tbb::auto_partitioner(), stopper);
		}

		if (stopper.is_group_execution_cancelled())
			break;
	}
	target->minval = -temp.maxval;
	target->maxval = temp.maxval;
	target->roi = temp.roi;

	STOPWATCH_PRINT(s, "Gradient TBB")

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
		delete current->swap(target);
		return true;
	}
}

void GradientTbb::Log::operator()(const tbb::blocked_range<size_t> &r) const
{
	for (size_t i = r.begin(); i != r.end(); ++i) {
		cv::log(source.bands[i], target.bands[i]);
		//target.bands[i] = cv::max(target.bands[i], 0.);
		cv::max(target.bands[i], 0., target.bands[i]);
	}
}

void GradientTbb::Grad::operator()(const tbb::blocked_range<size_t> &r) const
{
	for (size_t i = r.begin(); i != r.end(); ++i) {
		//target.bands[i] = source.bands[i + 1] - source.bands[i];
		cv::subtract(source.bands[i + 1], source.bands[i], target.bands[i]);
		if (!source.meta[i].empty && !source.meta[i + 1].empty) {
			target.meta[i] = multi_img::BandDesc(
				source.meta[i].center, source.meta[i + 1].center);
		}
	}
}

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
		delete current->swap(target);
		return true;
	}
}

bool PcaTbb::run() 
{
	cv::Mat_<multi_img::Value> pixels(
		(*source)->size(), (*source)->width * (*source)->height);
	Pixels computePixels(**source, pixels);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, (*source)->size()), 
		computePixels, tbb::auto_partitioner(), stopper);

	cv::PCA pca(pixels, cv::noArray(), CV_PCA_DATA_AS_COL, (int)components);

	multi_img *target = new multi_img(
		(*source)->height, (*source)->width, pca.eigenvectors.rows);
	Projection computeProjection(pixels, *target, pca);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, target->pixels.size()), 
		computeProjection, tbb::auto_partitioner(), stopper);

	ApplyCache applyCache(*target);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, target->size()), 
		applyCache, tbb::auto_partitioner(), stopper);

	DetermineRange determineRange(*target);
	tbb::parallel_reduce(tbb::blocked_range<size_t>(0, target->size()), 
		determineRange, tbb::auto_partitioner(), stopper);

	if (!stopper.is_group_execution_cancelled()) {
		target->dirty.setTo(0);
		target->anydirt = false;
		target->minval = determineRange.GetMin();
		target->maxval = determineRange.GetMax();
		target->roi = (*source)->roi;
	}

	if (!includecache)
		target->resetPixels();

	if (stopper.is_group_execution_cancelled()) {
		delete target;
		return false;
	} else {
		SharedDataSwapLock lock(current->mutex);
		delete current->swap(target);
		return true;
	}
}

void PcaTbb::Pixels::operator()(const tbb::blocked_range<size_t> &r) const
{
	for (size_t d = r.begin(); d != r.end(); ++d) {
		multi_img::Band &src = source.bands[d];
		multi_img::Band::const_iterator it; size_t i;
		for (it = src.begin(), i = 0; it != src.end(); ++it, ++i)
			target(d, i) = *it;
	}
}

void PcaTbb::Projection::operator()(const tbb::blocked_range<size_t> &r) const
{
	for (size_t i = r.begin(); i != r.end(); ++i) {
		cv::Mat_<multi_img::Value> input = source.col(i);
		cv::Mat_<multi_img::Value> output(target.pixels[i], false);
		pca.project(input, output);
	}
}

bool DataRangeTbb::run() 
{
	vole::Stopwatch s;

	DetermineRange determineRange(**multi);
	tbb::parallel_reduce(tbb::blocked_range<size_t>(0, (*multi)->size()), 
		determineRange, tbb::auto_partitioner(), stopper);

	STOPWATCH_PRINT(s, "DataRange TBB")

	if (!stopper.is_group_execution_cancelled()) {
		SharedDataSwapLock lock(range->mutex);
		(*range)->min = determineRange.GetMin();
		(*range)->max = determineRange.GetMax();
		return true;
	} else {
		return false;
	}
}

bool DataRangeCuda::run() 
{
	vole::Stopwatch s;

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

bool ClampTbb::run() 
{
	multi_img *source = &**image;
	assert(0 != source);

	multi_img *target = new multi_img(source->height, source->width, source->size());
	target->roi = source->roi;
	target->meta = source->meta;
	target->minval = (*minmax)->minval;
	target->maxval = (*minmax)->maxval;

	vole::Stopwatch s;

	Clamp computeClamp(*source, *target);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, target->size()), 
		computeClamp, tbb::auto_partitioner(), stopper);

	STOPWATCH_PRINT(s, "Clamp TBB")

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
		delete image->swap(target);
		return true;
	}
}

void ClampTbb::Clamp::operator()(const tbb::blocked_range<size_t> &r) const
{
	for (size_t d = r.begin(); d != r.end(); ++d) {
		multi_img::Band &src = source.bands[d];
		multi_img::Band &tgt = target.bands[d];
		cv::max(src, target.minval, tgt);
		cv::min(tgt, target.maxval, tgt);
	}
}

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
		delete image->swap(target);
		return true;
	}
}

bool IlluminantTbb::run() 
{
	multi_img *source = &**multi;
	assert(0 != source);
	multi_img *target = new multi_img(source->height, source->width, source->size());
	target->roi = source->roi;
	target->meta = source->meta;
	target->minval = source->minval;
	target->maxval = source->maxval;

	vole::Stopwatch s;

	Illumination computeIllumination(*source, *target, il, remove);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, target->size()), 
		computeIllumination, tbb::auto_partitioner(), stopper);

	STOPWATCH_PRINT(s, "Illuminant TBB")

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
		delete multi->swap(target);
		return true;
	}
}

void IlluminantTbb::Illumination::operator()(const tbb::blocked_range<size_t> &r) const
{
	if (remove) {
		for (size_t d = r.begin(); d != r.end(); ++d) {
			multi_img::Band &src = source.bands[d];
			multi_img::Band &tgt = target.bands[d];
			tgt = src / (multi_img::Value)il.at(source.meta[d].center);
		}
	} else {
		for (size_t d = r.begin(); d != r.end(); ++d) {
			multi_img::Band &src = source.bands[d];
			multi_img::Band &tgt = target.bands[d];
			tgt = src * (multi_img::Value)il.at(source.meta[d].center);
		}
	}
}

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
		delete multi->swap(target);
		return true;
	}
}

}

#endif
