#include "multi_img_tbb.h"
#include "cieobserver.h"

#include <multi_img.h>
#include <multi_img/illuminant.h>

#include <opencv2/core/core.hpp>
#include <tbb/parallel_for.h>
#include <cstddef>
#include <algorithm>

void RebuildPixels::operator()(const tbb::blocked_range<size_t> &r) const
{
	for (size_t d = r.begin(); d != r.end(); ++d) {
		multi_img::Band &src = multi.bands[d];
		if (src.empty()) {
			return;
		}
		multi_img::Band::const_iterator it; size_t i;
		for (it = src.begin(), i = 0; it != src.end(); ++it, ++i)
			multi.pixels[i][d] = *it;
	}
}

void RebuildPixels::operator()(const tbb::blocked_range2d<int> &r) const
{
	for (int row = r.rows().begin(); row != r.rows().end(); ++row) {
		int loc = row * multi.width;
		for (int col = r.cols().begin(); col != r.cols().end(); ++col) {
			for (size_t d = 0; d < multi.size(); ++d)
				multi.pixels[loc + col][d] = multi.bands[d](row, col);
		}
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

void ApplyCache::operator()(const tbb::blocked_range2d<int> &r) const
{
	for (int row = r.rows().begin(); row != r.rows().end(); ++row) {
		int loc = row * multi.width;
		for (int col = r.cols().begin(); col != r.cols().end(); ++col) {
			for (size_t d = 0; d < multi.size(); ++d)
				multi.bands[d](row, col) = multi.pixels[loc + col][d];
		}
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


void Xyz::operator()(const tbb::blocked_range2d<int> &r) const
{
	float intensity;
	float factor = 1.f / multi.maxval;
	__m128 cie_reg = _mm_setr_ps(CIEObserver::x[cie], 0.f,
	                             CIEObserver::y[cie], CIEObserver::z[cie]);

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
			__m128 res_reg = _mm_add_ps(v_reg,
			                            _mm_mul_ps(cie_reg,
			                                       _mm_load1_ps(&intensity)));
			_mm_storeh_pi((__m64*)&v[1], res_reg);
			_mm_store_ss(&v[0], res_reg);
		}
	}
}


void Bgr::operator()(const tbb::blocked_range2d<int> &r) const
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



void Grad::operator ()(const tbb::blocked_range<size_t> &r) const
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


void Log::operator ()(const tbb::blocked_range<size_t> &r) const
{
	{
		for (size_t i = r.begin(); i != r.end(); ++i) {
			cv::log(source.bands[i], target.bands[i]);
			//target.bands[i] = cv::max(target.bands[i], 0.);
			cv::max(target.bands[i], 0., target.bands[i]);
		}
	}
}

void NormL2::operator()(const tbb::blocked_range2d<int> &r) const
{
	for (int row = r.rows().begin(); row != r.rows().end(); ++row) {
		for (int col = r.cols().begin(); col != r.cols().end(); ++col) {
			cv::Mat_<multi_img::Value> src(
						source.pixels[row * source.width + col], false);
			cv::Mat_<multi_img::Value> dst(
						target.pixels[row * source.width + col], false);
			double n = cv::norm(src, cv::NORM_L2);
			if (n == 0.)
				n = 1.;
			dst = src / n;
		}
	}
}

void Clamp::operator ()(const tbb::blocked_range<size_t> &r) const
{
	for (size_t d = r.begin(); d != r.end(); ++d) {
		multi_img::Band &src = source.bands[d];
		multi_img::Band &tgt = target.bands[d];
		cv::max(src, target.minval, tgt);
		cv::min(tgt, target.maxval, tgt);
	}
}

void Illumination::operator ()(const tbb::blocked_range<size_t> &r) const
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


void PcaProjection::operator ()(const tbb::blocked_range<size_t> &r) const
{
	for (size_t i = r.begin(); i != r.end(); ++i) {
		cv::Mat_<multi_img::Value> input = source.col(i);
		cv::Mat_<multi_img::Value> output(target.pixels[i], false);
		pca.project(input, output);
	}
}


void MultiImg2BandMat::operator ()(const tbb::blocked_range<size_t> &r) const
{
	for (size_t d = r.begin(); d != r.end(); ++d) {
		multi_img::Band &src = source.bands[d];
		multi_img::Band::const_iterator it; size_t i;
		for (it = src.begin(), i = 0; it != src.end(); ++it, ++i)
			target(d, i) = *it;
	}
}

void Resize::operator()(const tbb::blocked_range2d<int> &r) const
{
	for (int row = r.rows().begin(); row != r.rows().end(); ++row) {
		for (int col = r.cols().begin(); col != r.cols().end(); ++col) {
			cv::Mat_<multi_img::Value> src(
			            source.pixels[row * source.width + col], false);
			cv::Mat_<multi_img::Value> dst(
			            target.pixels[row * source.width + col], false);
			cv::resize(src, dst, cv::Size(1, newsize));
		}
	}
}
