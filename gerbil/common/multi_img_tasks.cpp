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

//#define STOPWATCH_PRINT(stopwatch, message) stopwatch.print_reset(message);
#define STOPWATCH_PRINT(stopwatch, message)

namespace CIEObserver {
	extern float x[];
	extern float y[];
	extern float z[];
}

namespace MultiImg {

namespace Auxiliary {

int RectComplement(int width, int height, cv::Rect r, std::vector<cv::Rect> &result)
{
	result.push_back(cv::Rect(0, 0, r.x, r.y));
	result.push_back(cv::Rect(r.x, 0, r.width, r.y));
	result.push_back(cv::Rect(r.x + r.width, 0, width - r.width - r.x, r.y));
	result.push_back(cv::Rect(0, r.y, r.x, r.height));
	result.push_back(cv::Rect(r.x + r.width, r.y, width - r.width - r.x, r.height));
	result.push_back(cv::Rect(0, r.y + r.height, r.x, height - r.height - r.y));
	result.push_back(cv::Rect(r.x, r.y + r.height, r.width, height - r.height - r.y));
	result.push_back(cv::Rect(r.x + r.width, r.y + r.height, width - r.width - r.x, height - r.height - r.y));

	int area = 0;
	std::vector<cv::Rect>::iterator it;
	for (it = result.begin(); it != result.end(); ++it) {
		area += (it->width * it->height);
	}

	return area;
}

}

namespace CommonTbb {

void CommonTbb::RebuildPixels::operator()(const tbb::blocked_range<size_t> &r) const
{
	for (size_t d = r.begin(); d != r.end(); ++d) {
		multi_img::Band &src = multi.bands[d];
		multi_img::BandConstIt it; size_t i;
		for (it = src.begin(), i = 0; it != src.end(); ++it, ++i)
			multi.pixels[i][d] = *it;
	}
}

void CommonTbb::ApplyCache::operator()(const tbb::blocked_range<size_t> &r) const
{
	for (size_t d = r.begin(); d != r.end(); ++d) {
		multi_img::Band &dst = multi.bands[d];
		multi_img::BandIt it; size_t i;
		for (it = dst.begin(), i = 0; it != dst.end(); ++it, ++i)
			*it = multi.pixels[i][d];
	}
}

void CommonTbb::DetermineRange::operator()(const tbb::blocked_range<size_t> &r)
{
	double tmp1, tmp2;
	for (size_t d = r.begin(); d != r.end(); ++d) {
		cv::minMaxLoc(multi.bands[d], &tmp1, &tmp2);
		min = std::min<multi_img::Value>(min, (multi_img::Value)tmp1);
		max = std::max<multi_img::Value>(max, (multi_img::Value)tmp2);
	}
}

void CommonTbb::DetermineRange::join(DetermineRange &toJoin)
{
	if (toJoin.min < min)
		min = toJoin.min;
	if (toJoin.max > max)
		max = toJoin.max;
}

}

bool ScopeImage::run() 
{
	multi_img *target =  new multi_img(**full, targetRoi);
	SharedDataSwapLock lock(scoped->mutex);
	delete scoped->swap(target);
	return true;
}

bool BgrSerial::run() 
{
	cv::Mat_<cv::Vec3f> *newBgr = new cv::Mat_<cv::Vec3f>();
	*newBgr = (*multi)->bgr(); 
	SharedDataSwapLock lock(bgr->mutex);
	delete bgr->swap(newBgr);
	return true;
}

bool BgrTbb::run() 
{
	cv::Mat_<cv::Vec3f> xyz((*multi)->height, (*multi)->width, 0.);
	float greensum = 0.f;
	for (size_t i = 0; i < (*multi)->size(); ++i) {
		int idx = ((int)((*multi)->meta[i].center + 0.5f) - 360) / 5;
		if (idx < 0 || idx > 94)
			continue;

		multi_img::Band band;
		(*multi)->getBand(i, band);
		Xyz computeXyz(**multi, xyz, band, idx);
		tbb::parallel_for(tbb::blocked_range2d<int>(0, xyz.rows, 0, xyz.cols), 
			computeXyz, tbb::auto_partitioner(), stopper);

		greensum += CIEObserver::y[idx];

		if (stopper.is_group_execution_cancelled())
			break;
	}

	if (greensum == 0.f)
		greensum = 1.f;

	cv::Mat_<cv::Vec3f> *newBgr = new cv::Mat_<cv::Vec3f>((*multi)->height, (*multi)->width);
	Bgr computeBgr(**multi, xyz, *newBgr, greensum);
	tbb::parallel_for(tbb::blocked_range2d<int>(0, newBgr->rows, 0, newBgr->cols), 
		computeBgr, tbb::auto_partitioner(), stopper);

	if (stopper.is_group_execution_cancelled()) {
		delete newBgr;
		return false;
	} else {
		SharedDataSwapLock lock(bgr->mutex);
		delete bgr->swap(newBgr);
		return true;
	}
}

void BgrTbb::Xyz::operator()(const tbb::blocked_range2d<int> &r) const
{
	float intensity;
	float factor = 1.f / multi.maxval;
	__m128 cie_reg = _mm_setr_ps(CIEObserver::x[cie], 0.f, CIEObserver::y[cie], CIEObserver::z[cie]);

	for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
		for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
			/*
			cv::Vec3f &v = xyz(i, j);
			float intensity = band(i, j) * 1.f / multi.maxval;
			v[0] += CIEObserver::x[cie] * intensity;
			v[1] += CIEObserver::y[cie] * intensity;
			v[2] += CIEObserver::z[cie] * intensity;
			*/
			
			cv::Vec3f &v = xyz(i, j);
			intensity = band(i, j) * factor;
			__m128 v_reg = _mm_loadh_pi(_mm_load_ss(&v[0]), (__m64*)&v[1]);
			__m128 res_reg = _mm_add_ps(v_reg, _mm_mul_ps(cie_reg, _mm_load1_ps(&intensity)));
			_mm_storeh_pi((__m64*)&v[1], res_reg);
			_mm_store_ss(&v[0], res_reg);
		}
	}
}

void BgrTbb::Bgr::operator()(const tbb::blocked_range2d<int> &r) const
{
	for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
		for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
			/*
			cv::Vec3f &vs = xyz(i, j);
			cv::Vec3f &vd = bgr(i, j);
			for (unsigned int j = 0; j < 3; ++j)
				vs[j] /= greensum;
			multi.xyz2bgr(vs, vd);
			*/

			cv::Vec3f &vs = xyz(i, j);
			cv::Vec3f &vd = bgr(i, j);
			__m128 vs_reg = _mm_loadh_pi(_mm_load_ss(&vs[0]), (__m64*)&vs[1]);
			__m128 res_reg = _mm_div_ps(vs_reg, _mm_load1_ps(&greensum));
			_mm_storeh_pi((__m64*)&vs[1], res_reg);
			_mm_store_ss(&vs[0], res_reg);
			multi_img::xyz2bgr(vs, vd);
		}
	}
}

#ifdef WITH_QT
bool Band2QImageTbb::run() 
{
	if (band >= (*multi)->size()) {
		return false;
	}

	multi_img::Band &source = (*multi)->bands[band];
	QImage *target = new QImage(source.cols, source.rows, QImage::Format_ARGB32);
	Conversion computeConversion(source, *target, (*multi)->minval, (*multi)->maxval);
	tbb::parallel_for(tbb::blocked_range2d<int>(0, source.rows, 0, source.cols), 
		computeConversion, tbb::auto_partitioner(), stopper);

	if (stopper.is_group_execution_cancelled()) {
		delete target;
		return false;
	} else {
		SharedDataSwapLock lock(image->mutex);
		delete image->swap(target);
		return true;
	}
}

void Band2QImageTbb::Conversion::operator()(const tbb::blocked_range2d<int> &r) const
{
	multi_img::Value scale = 255.0 / (maxval - minval);
	for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
		const multi_img::Value *srcrow = band[y];
		QRgb *destrow = (QRgb*)image.scanLine(y);
		for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
			unsigned int color = (srcrow[x] - minval) * scale;
			destrow[x] = qRgba(color, color, color, 255);
		}
	}
}
#endif

bool RescaleTbb::run() 
{
	multi_img *temp = new multi_img(**source, 
		cv::Rect(0, 0, (*source)->width, (*source)->height));
	temp->roi = (*source)->roi;
	CommonTbb::RebuildPixels rebuildPixels(*temp);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, temp->size()), 
		rebuildPixels, tbb::auto_partitioner(), stopper);
	temp->dirty.setTo(0);
	temp->anydirt = false;

	multi_img *target = NULL;
	if (newsize != temp->size()) {

		target = new multi_img((*source)->height, (*source)->width, newsize);
		target->minval = temp->minval;
		target->maxval = temp->maxval;
		target->roi = temp->roi;
		Resize computeResize(*temp, *target, newsize);
		tbb::parallel_for(tbb::blocked_range2d<int>(0, temp->height, 0, temp->width), 
			computeResize, tbb::auto_partitioner(), stopper);

		CommonTbb::ApplyCache applyCache(*target);
		tbb::parallel_for(tbb::blocked_range<size_t>(0, target->size()), 
			applyCache, tbb::auto_partitioner(), stopper);
		target->dirty.setTo(0);
		target->anydirt = false;

		if (!stopper.is_group_execution_cancelled()) {
			cv::Mat_<float> tmpmeta1(cv::Size(temp->meta.size(), 1)), tmpmeta2;
			std::vector<multi_img::BandDesc>::const_iterator it;
			unsigned int i;
			for (it = temp->meta.begin(), i = 0; it != temp->meta.end(); it++, i++) {
				tmpmeta1(0, i) = it->center;
			}
			cv::resize(tmpmeta1, tmpmeta2, cv::Size(newsize, 1));
			for (size_t b = 0; b < newsize; b++) {
				target->meta[b] = multi_img::BandDesc(tmpmeta2(0, b));
			}
		}

		delete temp;
		temp = NULL;
	
	} else {
		target = temp;
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

void RescaleTbb::Resize::operator()(const tbb::blocked_range2d<int> &r) const
{
	for (int row = r.rows().begin(); row != r.rows().end(); ++row) {
		for (int col = r.cols().begin(); col != r.cols().end(); ++col) {
			cv::Mat_<multi_img::Value> src(source.pixels[row * source.width + col], false);
			cv::Mat_<multi_img::Value> dst(target.pixels[row * source.width + col], false);
			cv::resize(src, dst, cv::Size(1, newsize));
		}
	}
}

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
	Auxiliary::RectComplement((*source)->width, (*source)->height, copySrc, calc);

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
		CommonTbb::RebuildPixels rebuildPixels(*target);
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
	Auxiliary::RectComplement((*source)->width, (*source)->height, copySrc, calc);

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
		CommonTbb::RebuildPixels rebuildPixels(*target);
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

	CommonTbb::ApplyCache applyCache(*target);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, target->size()), 
		applyCache, tbb::auto_partitioner(), stopper);

	CommonTbb::DetermineRange determineRange(*target);
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
		multi_img::BandConstIt it; size_t i;
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

	CommonTbb::DetermineRange determineRange(**multi);
	tbb::parallel_reduce(tbb::blocked_range<size_t>(0, (*multi)->size()), 
		determineRange, tbb::auto_partitioner(), stopper);

	STOPWATCH_PRINT(s, "DataRange TBB")

	if (!stopper.is_group_execution_cancelled()) {
		SharedDataSwapLock lock(range->mutex);
		(*range)->first = determineRange.GetMin();
		(*range)->second = determineRange.GetMax();
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
		(*range)->first = min;
		(*range)->second = max;
		return true;
	} else {
		return false;
	}
}

bool ClampTbb::run() 
{
	multi_img *source;
	if(use_multi_img_base)
		source = dynamic_cast<multi_img*>(&(**multi_base));
	else
		source = &(**multi_full);
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
		CommonTbb::RebuildPixels rebuildPixels(*target);
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
		if(use_multi_img_base) {
			SharedDataSwapLock lock(multi_base->mutex);
			delete multi_base->swap(target);
		} else {
			SharedDataSwapLock lock(multi_full->mutex);
			delete multi_full->swap(target);
		}
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
	// BUG
	// There is no reasonable way to actually get the pointer to the multi_img
	// from SharedData.
	// multi_img_ptr x;
	// typeof(*x) == SharedData<multi_img> != multi_img*
	// typeof(**x) == multi_img
	// typeof(&(**x)) == multi_img*    [doh!]

	multi_img *source;
	if(use_multi_img_base)
		source = dynamic_cast<multi_img*>(&(**multi_base));
	else
		source = &(**multi_full);

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
		CommonTbb::RebuildPixels rebuildPixels(*target);
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
		if(use_multi_img_base) {
			SharedDataSwapLock lock(multi_base->mutex);
			delete multi_base->swap(target);
		} else {
			SharedDataSwapLock lock(multi_full->mutex);
			delete multi_full->swap(target);
		}
		return true;
	}
}

bool IlluminantTbb::run() 
{
	multi_img *source = dynamic_cast<multi_img*>(&(**multi));
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
		CommonTbb::RebuildPixels rebuildPixels(*target);
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
	multi_img *source = dynamic_cast<multi_img*>(&(**multi));
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
		CommonTbb::RebuildPixels rebuildPixels(*target);
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
