/*
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include <multi_img.h>
#include "illuminant.h"
#include "cieobserver.h"

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>

multi_img multi_img::spec_gradient() const
{
	multi_img ret(size() - 1);
	// data format of output
	ret.minval = -maxval;
	ret.maxval =  maxval;
	ret.width = width; ret.height = height;

	for (unsigned int i = 0; i < size()-1; ++i) {
		ret.bands[i] = (bands[i+1] - bands[i]);
		if (!meta[i].empty && !meta[i+1].empty)
			ret.meta[i] = BandDesc(meta[i].center, meta[i+1].center);
	}
	ret.resetPixels();

	return ret;
}

multi_img multi_img::spec_rescale(unsigned int newsize) const
{
	std::cerr << "multi_img: spec_rescale(" << newsize << ")" << std::endl;
	assert(newsize <= this->size());

	if (newsize == this->size())
		return multi_img(*this);

	multi_img ret(height, width, newsize);
	ret.minval = minval;
	ret.maxval = maxval;

	rebuildPixels();
	for (int row = 0; row < height; ++row) {
		for (int col = 0; col < width; ++col) {
			/// delegate resizing to opencv, using mat headers over vectors
			cv::Mat_<Value> src(pixels[row*width + col]),
			                dst(ret.pixels[row*width + col]);
			cv::resize(src, dst, cv::Size(1, newsize));
		}
	}

	/// ret: write back pixel cache into bands
	ret.applyCache();

	/// interpolate wavelength metadata accordingly
	{
		cv::Mat_<float> tmpmeta1(cv::Size(meta.size(), 1)), tmpmeta2;
		std::vector<BandDesc>::const_iterator it;
		unsigned int i;
		for (it = meta.begin(), i = 0; it != meta.end(); it++, i++) {
			tmpmeta1(0, i) = it->center;
		}
		cv::resize(tmpmeta1, tmpmeta2, cv::Size(newsize, 1));
		for (size_t b = 0; b < newsize; b++) {
			ret.meta[b] = BandDesc(tmpmeta2(0, b));
		}
	}

	return ret;
}

void multi_img::pixel2xyz(const Pixel &p, cv::Vec3f &v,
	size_t dim, const std::vector<BandDesc> &meta, Value maxval)
{
	int idx;
	float intensity;
	float greensum = 0.;
	float factor = 1.f / maxval;
	for (size_t i = 0; i < dim; ++i) {
		idx = ((int)(meta[i].center + 0.5f) - 360) / 5;
		if (idx < 0 || idx > 94)
			continue;
		intensity = p[i] * factor;

		/*
		v[0] += CIEObserver::x[idx] * intensity;
		v[1] += CIEObserver::y[idx] * intensity;
		v[2] += CIEObserver::z[idx] * intensity;
		*/

		__m128 cie_reg = _mm_setr_ps(CIEObserver::x[idx], 0.f, CIEObserver::y[idx], CIEObserver::z[idx]);
		__m128 v_reg = _mm_loadh_pi(_mm_load_ss(&v[0]), (__m64*)&v[1]);
		__m128 res_reg = _mm_add_ps(v_reg, _mm_mul_ps(cie_reg, _mm_load1_ps(&intensity)));
		_mm_storeh_pi((__m64*)&v[1], res_reg);
		_mm_store_ss(&v[0], res_reg);

		greensum += CIEObserver::y[idx];
	}
	if (greensum == 0.f)	// we didn't collect valuable data.
		return;

	/*
	for (unsigned int i = 0; i < 3; ++i)
		v[i] /= greensum;
	*/

	__m128 v_reg = _mm_loadh_pi(_mm_load_ss(&v[0]), (__m64*)&v[1]);
	__m128 res_reg = _mm_div_ps(v_reg, _mm_load1_ps(&greensum));
	_mm_storeh_pi((__m64*)&v[1], res_reg);
	_mm_store_ss(&v[0], res_reg);
}

void multi_img::pixel2xyz(const Pixel &p, cv::Vec3f &v) const
{
	pixel2xyz(p, v, p.size(), meta, maxval);
}

void multi_img::xyz2bgr(const cv::Vec3f &vs, cv::Vec3f &vd)
{
	/* Inverse M for sRGB, D65 */
	/*
	vd[2] =  3.2404542f * vs[0] + -1.5371385f * vs[1] + -0.4985314f * vs[2];
	vd[1] = -0.9692660f * vs[0] +  1.8760108f * vs[1] +  0.0415560f * vs[2];
	vd[0] =  0.0556434f * vs[0] + -0.2040259f * vs[1] +  1.0572252f * vs[2];
	*/

	_mm_setr_ps(0.0556434f, 0.f, -0.9692660f, 3.2404542f);
	_mm_setr_ps(-0.2040259f, 0.f, 1.8760108f, -1.5371385f);
	_mm_setr_ps(1.0572252f, 0.f, 0.0415560f, -0.4985314f);
	_mm_load1_ps(&vs[0]);
	_mm_load1_ps(&vs[1]);
	_mm_load1_ps(&vs[2]);

	__m128 vd_reg = _mm_add_ps(
		_mm_mul_ps(
			_mm_setr_ps(0.0556434f, 0.f, -0.9692660f, 3.2404542f),
			_mm_load1_ps(&vs[0])),
		_mm_add_ps(
			_mm_mul_ps(
				_mm_setr_ps(-0.2040259f, 0.f, 1.8760108f, -1.5371385f),
				_mm_load1_ps(&vs[1])),
			_mm_mul_ps(
				_mm_setr_ps(1.0572252f, 0.f, 0.0415560f, -0.4985314f),
				_mm_load1_ps(&vs[2]))));

	__m128 sanitized_reg = _mm_min_ps(
		_mm_max_ps(vd_reg, _mm_setzero_ps()),
		_mm_setr_ps(1.f, 1.f, 1.f, 1.f));

	_mm_storeh_pi((__m64*)&vd[1], sanitized_reg);
	_mm_store_ss(&vd[0], sanitized_reg);

	/* default Gamma correction for sRGB */
	float gamma = 1.f/2.4f;
	for (unsigned int i = 0; i < 3; ++i) {
		// sanitize first
		//vd[i] = std::max<float>(vd[i], 0.f);
		//vd[i] = std::min<float>(vd[i], 1.f);
		// now apply gamma the sRGB way
		if (vd[i] > 0.0031308f)
			vd[i] = 1.055f * std::pow(vd[i], gamma) - 0.055f;
		else
			vd[i] = 12.92f * vd[i];
	}
}

cv::Mat_<cv::Vec3f> multi_img::bgr() const
{
	cv::Mat_<cv::Vec3f> xyz(height, width, 0.);
	float greensum = 0.f;
	for (size_t i = 0; i < size(); ++i) {
		int idx = ((int)(meta[i].center + 0.5f) - 360) / 5;
		if (idx < 0 || idx > 94)
			continue;
		Band::const_iterator src = bands[i].begin();
		cv::Mat_<cv::Vec3f>::iterator dst = xyz.begin();
		for (; dst != xyz.end(); ++src, ++dst) {
			cv::Vec3f &v = *dst;
			float intensity = (*src) * 1.f/maxval;
			v[0] += CIEObserver::x[idx] * intensity;
			v[1] += CIEObserver::y[idx] * intensity;
			v[2] += CIEObserver::z[idx] * intensity;
		}
		greensum += CIEObserver::y[idx];
	}

	if (greensum == 0.f)
		greensum = 1.f;

	cv::Mat_<cv::Vec3f> bgr(height, width);
	cv::Mat_<cv::Vec3f>::iterator src = xyz.begin();
	cv::Mat_<cv::Vec3f>::iterator dst = bgr.begin();
	for (; dst != bgr.end(); ++src, ++dst) {
		cv::Vec3f &vs = *src;
		cv::Vec3f &vd = *dst;
		for (unsigned int i = 0; i < 3; ++i)
			vs[i] /= greensum;
		xyz2bgr(vs, vd);
	}
	return bgr;
}

cv::Vec3f multi_img::bgr(const Pixel &p) const
{
	cv::Vec3f xyz, ret;
	pixel2xyz(p, xyz);
	xyz2bgr(xyz, ret);
	return ret;
}

cv::Vec3f multi_img::bgr(const Pixel &p,
	const std::vector<BandDesc> &meta, Value maxval)
{
	cv::Vec3f xyz, ret;
	pixel2xyz(p, xyz, p.size(), meta, maxval);
	xyz2bgr(xyz, ret);
	return ret;
}

void multi_img::apply_illuminant(const Illuminant& il, bool remove)
{
	if (remove) {
		for (size_t i = 0; i < size(); ++i)
			bands[i] /= (Value)il.at(meta[i].center);
		// if meta was not set, center is 0., and il.at() will throw assert :)
	} else {
		for (size_t i = 0; i < size(); ++i)
			bands[i] *= (Value)il.at(meta[i].center);
	}
	// cache became invalid
	resetPixels();
}

std::vector<multi_img::Value> multi_img::getIllumCoeff(const Illuminant & il) const
{
	std::vector<Value> ret(size());
	for (size_t i = 0; i < size(); ++i)
		ret[i] = (Value)il.at(meta[i].center);
	return ret;
}
