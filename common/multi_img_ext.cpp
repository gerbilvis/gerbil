#include <multi_img.h>
#include "illuminant.h"

multi_img multi_img::spec_gradient() const
{
	multi_img ret(size() - 1);
	// data format of output
	ret.minval = -maxval/2.;
	ret.maxval =  maxval/2.;
	ret.width = width; ret.height = height;

	for (unsigned int i = 0; i < size()-1; ++i) {
		ret.bands[i] = (bands[i+1] - bands[i]);
		if (!meta[i].empty && !meta[i+1].empty)
			ret.meta[i] = BandDesc(meta[i].center, meta[i+1].center);
	}
	ret.resetPixels();
	return ret;
}

namespace CIEObserver {	// 10 degree 1964 CIE observer coefficients
	multi_img::Value x[] = { 1.222E-07, 9.1927E-07, 5.9586E-06, 3.3266E-05, 0.000159952, 0.00066244, 0.0023616, 0.0072423, 0.0191097, 0.0434, 0.084736, 0.140638, 0.204492, 0.264737, 0.314679, 0.357719, 0.383734, 0.386726, 0.370702, 0.342957, 0.302273, 0.254085, 0.195618, 0.132349, 0.080507, 0.041072, 0.016172, 0.005132, 0.003816, 0.015444, 0.037465, 0.071358, 0.117749, 0.172953, 0.236491, 0.304213, 0.376772, 0.451584, 0.529826, 0.616053, 0.705224, 0.793832, 0.878655, 0.951162, 1.01416, 1.0743, 1.11852, 1.1343, 1.12399, 1.0891, 1.03048, 0.95074, 0.856297, 0.75493, 0.647467, 0.53511, 0.431567, 0.34369, 0.268329, 0.2043, 0.152568, 0.11221, 0.0812606, 0.05793, 0.0408508, 0.028623, 0.0199413, 0.013842, 0.00957688, 0.0066052, 0.00455263, 0.0031447, 0.00217496, 0.0015057, 0.00104476, 0.00072745, 0.000508258, 0.00035638, 0.000250969, 0.00017773, 0.00012639, 9.0151E-05, 6.45258E-05, 4.6339E-05, 3.34117E-05, 2.4209E-05, 1.76115E-05, 1.2855E-05, 9.41363E-06, 6.913E-06, 5.09347E-06, 3.7671E-06, 2.79531E-06, 2.082E-06, 1.55314E-06 };
	multi_img::Value y[] = { 1.3398E-08, 1.0065E-07, 6.511E-07, 3.625E-06, 1.7364E-05, 7.156E-05, 0.0002534, 0.0007685, 0.0020044, 0.004509, 0.008756, 0.014456, 0.021391, 0.029497, 0.038676, 0.049602, 0.062077, 0.074704, 0.089456, 0.106256, 0.128201, 0.152761, 0.18519, 0.21994, 0.253589, 0.297665, 0.339133, 0.395379, 0.460777, 0.53136, 0.606741, 0.68566, 0.761757, 0.82333, 0.875211, 0.92381, 0.961988, 0.9822, 0.991761, 0.99911, 0.99734, 0.98238, 0.955552, 0.915175, 0.868934, 0.825623, 0.777405, 0.720353, 0.658341, 0.593878, 0.527963, 0.461834, 0.398057, 0.339554, 0.283493, 0.228254, 0.179828, 0.140211, 0.107633, 0.081187, 0.060281, 0.044096, 0.0318004, 0.0226017, 0.0159051, 0.0111303, 0.0077488, 0.0053751, 0.00371774, 0.00256456, 0.00176847, 0.00122239, 0.00084619, 0.00058644, 0.00040741, 0.000284041, 0.00019873, 0.00013955, 9.8428E-05, 6.9819E-05, 4.9737E-05, 3.55405E-05, 2.5486E-05, 1.83384E-05, 1.3249E-05, 9.6196E-06, 7.0128E-06, 5.1298E-06, 3.76473E-06, 2.77081E-06, 2.04613E-06, 1.51677E-06, 1.12809E-06, 8.4216E-07, 6.297E-07 };
	multi_img::Value z[] = { 5.35027E-07, 4.0283E-06, 2.61437E-05, 0.00014622, 0.000704776, 0.0029278, 0.0104822, 0.032344, 0.0860109, 0.19712, 0.389366, 0.65676, 0.972542, 1.2825, 1.55348, 1.7985, 1.96728, 2.0273, 1.9948, 1.9007, 1.74537, 1.5549, 1.31756, 1.0302, 0.772125, 0.5706, 0.415254, 0.302356, 0.218502, 0.159249, 0.112044, 0.082248, 0.060709, 0.04305, 0.030451, 0.020584, 0.013676, 0.007918, 0.003988, 0.001091, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
}

cv::Mat3f multi_img::rgb() const
{
	cv::Mat3f xyz(height, width, 0.);
	Value greensum = 0.;
	for (unsigned int i = 0; i < size(); ++i) {
		int idx = (meta[i].center - 360) / 5;
		if (idx < 0 || idx > 94)
			continue;
		BandConstIt src = bands[i].begin();
		cv::Mat3f::iterator dst = xyz.begin();
		for (; dst != xyz.end(); ++src, ++dst) {
			cv::Vec3f &v = *dst;
			Value intensity = (*src) * 1./maxval;
			v[0] += CIEObserver::x[idx] * intensity;
			v[1] += CIEObserver::y[idx] * intensity;
			v[2] += CIEObserver::z[idx] * intensity;
		}
		greensum += CIEObserver::y[idx];
	}

	cv::Mat3f rgb(height, width);
	cv::Mat3f::iterator src = xyz.begin();
	cv::Mat3f::iterator dst = rgb.begin();
	for (; dst != rgb.end(); ++src, ++dst) {
		cv::Vec3f &vs = *src;
		cv::Vec3f &vd = *dst;
		for (unsigned int i = 0; i < 3; ++i)
			vs[i] /= greensum;

		/* Inverse M for sRGB, D65 */
		vd[0] =  3.2404542 * vs[0] + -1.5371385 * vs[1] + -0.4985314 * vs[2];
		vd[1] = -0.9692660 * vs[0] +  1.8760108 * vs[1] +  0.0415560 * vs[2];
		vd[2] =  0.0556434 * vs[0] + -0.2040259 * vs[1] +  1.0572252 * vs[2];

		/* default Gamma correction for sRGB */
		float gamma = 1./2.4;
		for (unsigned int i = 0; i < 3; ++i) {
			if (vd[i] < 0.)	vd[i] = 0.;	if (vd[i] > 1.)	vd[i] = 1.;
			if (vd[i] > 0.0031308)
				vd[i] = 1.055*std::pow(vd[i], gamma) - 0.055;
			else
				vd[i] = 12.92*vd[i];
		}
	}
	return rgb;
}

void multi_img::apply_illuminant(const Illuminant& il, bool remove)
{
	if (remove) {
		for (unsigned int i = 0; i < size(); ++i)
			bands[i] /= il.at(meta[i].center);
		// if meta was not set, center is 0., and il.at() will throw assert :)
	} else {
		for (unsigned int i = 0; i < size(); ++i)
			bands[i] *= il.at(meta[i].center);
	}
	// cache became invalid
	resetPixels();
}

std::vector<multi_img::Value> multi_img::getIllumCoeff(const Illuminant & il) const
{
	std::vector<Value> ret(size());
	for (unsigned int i = 0; i < size(); ++i)
		ret[i] = il.at(meta[i].center);
	return ret;
}

