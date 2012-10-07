/*	
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include <multi_img.h>
#include "illuminant.h"

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

multi_img multi_img::spec_rescale(size_t newsize) const
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

namespace CIEObserver {	// 10 degree 1964 CIE observer coefficients
	float x[] = { 1.222E-07f, 9.1927E-07f, 5.9586E-06f, 3.3266E-05f, 0.000159952f, 0.00066244f, 0.0023616f, 0.0072423f, 0.0191097f, 0.0434f, 0.084736f, 0.140638f, 0.204492f, 0.264737f, 0.314679f, 0.357719f, 0.383734f, 0.386726f, 0.370702f, 0.342957f, 0.302273f, 0.254085f, 0.195618f, 0.132349f, 0.080507f, 0.041072f, 0.016172f, 0.005132f, 0.003816f, 0.015444f, 0.037465f, 0.071358f, 0.117749f, 0.172953f, 0.236491f, 0.304213f, 0.376772f, 0.451584f, 0.529826f, 0.616053f, 0.705224f, 0.793832f, 0.878655f, 0.951162f, 1.01416f, 1.0743f, 1.11852f, 1.1343f, 1.12399f, 1.0891f, 1.03048f, 0.95074f, 0.856297f, 0.75493f, 0.647467f, 0.53511f, 0.431567f, 0.34369f, 0.268329f, 0.2043f, 0.152568f, 0.11221f, 0.0812606f, 0.05793f, 0.0408508f, 0.028623f, 0.0199413f, 0.013842f, 0.00957688f, 0.0066052f, 0.00455263f, 0.0031447f, 0.00217496f, 0.0015057f, 0.00104476f, 0.00072745f, 0.000508258f, 0.00035638f, 0.000250969f, 0.00017773f, 0.00012639f, 9.0151E-05f, 6.45258E-05f, 4.6339E-05f, 3.34117E-05f, 2.4209E-05f, 1.76115E-05f, 1.2855E-05f, 9.41363E-06f, 6.913E-06f, 5.09347E-06f, 3.7671E-06f, 2.79531E-06f, 2.082E-06f, 1.55314E-06f };
	float y[] = { 1.3398E-08f, 1.0065E-07f, 6.511E-07f, 3.625E-06f, 1.7364E-05f, 7.156E-05f, 0.0002534f, 0.0007685f, 0.0020044f, 0.004509f, 0.008756f, 0.014456f, 0.021391f, 0.029497f, 0.038676f, 0.049602f, 0.062077f, 0.074704f, 0.089456f, 0.106256f, 0.128201f, 0.152761f, 0.18519f, 0.21994f, 0.253589f, 0.297665f, 0.339133f, 0.395379f, 0.460777f, 0.53136f, 0.606741f, 0.68566f, 0.761757f, 0.82333f, 0.875211f, 0.92381f, 0.961988f, 0.9822f, 0.991761f, 0.99911f, 0.99734f, 0.98238f, 0.955552f, 0.915175f, 0.868934f, 0.825623f, 0.777405f, 0.720353f, 0.658341f, 0.593878f, 0.527963f, 0.461834f, 0.398057f, 0.339554f, 0.283493f, 0.228254f, 0.179828f, 0.140211f, 0.107633f, 0.081187f, 0.060281f, 0.044096f, 0.0318004f, 0.0226017f, 0.0159051f, 0.0111303f, 0.0077488f, 0.0053751f, 0.00371774f, 0.00256456f, 0.00176847f, 0.00122239f, 0.00084619f, 0.00058644f, 0.00040741f, 0.000284041f, 0.00019873f, 0.00013955f, 9.8428E-05f, 6.9819E-05f, 4.9737E-05f, 3.55405E-05f, 2.5486E-05f, 1.83384E-05f, 1.3249E-05f, 9.6196E-06f, 7.0128E-06f, 5.1298E-06f, 3.76473E-06f, 2.77081E-06f, 2.04613E-06f, 1.51677E-06f, 1.12809E-06f, 8.4216E-07f, 6.297E-07f };
	float z[] = { 5.35027E-07f, 4.0283E-06f, 2.61437E-05f, 0.00014622f, 0.000704776f, 0.0029278f, 0.0104822f, 0.032344f, 0.0860109f, 0.19712f, 0.389366f, 0.65676f, 0.972542f, 1.2825f, 1.55348f, 1.7985f, 1.96728f, 2.0273f, 1.9948f, 1.9007f, 1.74537f, 1.5549f, 1.31756f, 1.0302f, 0.772125f, 0.5706f, 0.415254f, 0.302356f, 0.218502f, 0.159249f, 0.112044f, 0.082248f, 0.060709f, 0.04305f, 0.030451f, 0.020584f, 0.013676f, 0.007918f, 0.003988f, 0.001091f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
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
		BandConstIt src = bands[i].begin();
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

cv::Vec3f multi_img::bgr(const Pixel &p, size_t dim, 
	std::vector<BandDesc> &meta, Value maxval)
{
	cv::Vec3f xyz, ret;
	pixel2xyz(p, xyz, dim, meta, maxval);
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

