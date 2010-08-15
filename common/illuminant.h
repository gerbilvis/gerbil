#ifndef ILLUMINANT_H
#define ILLUMINANT_H

#include <multi_img.h>

class Illuminant {
public:
	Illuminant() {} // needed for std containers
	Illuminant(multi_img::Value temperature) : temp(temperature), weight(1.)
	{	// precalculate real value at 560nm to do norming
		norm560 = 100./at(560, true);
	}
	
	// calculate the weight such that all coeffs. in range <= 1.
	inline void calcWeight(int wl1, int wl2) {
		multi_img::Value maxv = 0.;
		for (int i = wl1; i <= wl2; i+=10)
			if (at(i) > maxv)	maxv = at(i);
		weight = 1./maxv;
	}
	
	inline multi_img::Value at(int wavelength, bool norm = false) const {
		assert(wavelength > 0);
		std::map<int, multi_img::Value>::iterator i = coeff.find(wavelength);
		if (i != coeff.end())
			return i->second * weight;
	
		// black body calculation
		double wl = wavelength * 0.000000001;
		double wl_5 = std::pow(wl, -5);
		double M = 0.00000000000000037418 * wl_5 /
				   (std::exp(0.014388 / (temp * wl)) - 1);
		if (!norm) {
			M = M * norm560;
			coeff[wavelength] = M;
		}
		return M * weight;
	}

private:
	mutable std::map<int, multi_img::Value> coeff;
	multi_img::Value temp;
	multi_img::Value norm560;
	multi_img::Value weight;
};

#endif
