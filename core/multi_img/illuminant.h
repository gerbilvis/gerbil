/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef ILLUMINANT_H
#define ILLUMINANT_H

#include <map>
#include <algorithm>
#include <cassert>
#include <cmath>

class Illuminant {
public:
	Illuminant() {} // needed for std containers
	Illuminant(double temperature) : temp(temperature), weight(1.)
	{	// precalculate real value at 560nm to do norming
		norm560 = 100./at(560, true);
	}

	// calculate the weight such that all coeffs. in range <= 1.
	inline void setNormalization(int wl1, int wl2) {
		double maxv = 0.;
		for (int i = wl1; i <= wl2; i+=10)
			maxv = std::max(at(i), maxv);

		weight = 1./maxv;
	}

	inline double at(int wavelength, bool norm = false) const {
		assert(wavelength > 0);
		std::map<int, double>::iterator i = coeff.find(wavelength);
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

	inline double at(float wavelength, bool norm = false) const {
		return at((int)(wavelength + 0.5f), norm);
	}

private:
	mutable std::map<int, double> coeff;
	double temp;
	double norm560;
	double weight;
};

#endif
