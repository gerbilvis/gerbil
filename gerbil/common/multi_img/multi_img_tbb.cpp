#include <cstddef>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <cv.h>

#include <multi_img.h>

#include "multi_img_tbb.h"
#include "cieobserver.h"


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


void Xyz::operator()(const tbb::blocked_range2d<int> &r) const
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
