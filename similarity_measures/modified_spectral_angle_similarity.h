/*	
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>,
	David Foehrweiser.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_MOD_SPEC_ANG_SIM_H
#define VOLE_MOD_SPEC_ANG_SIM_H

#include "similarity_measure.h"
#include <math.h>
#include <iostream>

#ifndef M_PI
#define M_PI	3.14159265358979323846264338327950288419716939937f
#endif

namespace vole {

template<typename T>
class ModifiedSpectralAngleSimilarity : public SimilarityMeasure<T> {

public:

	ModifiedSpectralAngleSimilarity() {}

	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);
	double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2);
};

template<typename T>
inline double ModifiedSpectralAngleSimilarity<T>::getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
{
	this->check(img1, img2);

	cv::MatConstIterator_<T> it1 = img1.begin(), it2 = img2.begin();
	double ret = 0.0f;
	double tt = 0.0f;
	double pp = 0.0f;
	double pt = 0.0f;


	for(; it1 < img1.end(); it1++, it2++) {
		tt += (*it1) * (*it1);
		pp += (*it2) * (*it2);
		pt += (*it1) * (*it2);
	}

	tt = std::sqrt(tt);
	pp = std::sqrt(pp);

	ret = std::acos(pt/(tt*pp));
	
	/* scale it [0,1]
		- not necessary for us
		- would break SIDSAM */
	// ret = 2.0f*ret/M_PI;

	return ret;
}

template<typename T>
inline double ModifiedSpectralAngleSimilarity<T>::getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2)
{
	this->check(v1, v2);

	typename std::vector<T>::const_iterator it1 = v1.begin(), it2 = v2.begin();

	double ret = 0.0f;
	
	double tt = 0.0f;
	double pp = 0.0f;
	double pt = 0.0f;


	for(; it1 < v1.end(); it1++, it2++) {
		tt += (*it1) * (*it1);
		pp += (*it2) * (*it2);
		pt += (*it1) * (*it2);
	}

	tt = std::sqrt(tt);
	pp = std::sqrt(pp);

	ret = std::acos(pt/(tt*pp));

	/* scale it [0,1]
		- not necessary for us
		- would break SIDSAM */
	// ret = 2.0f*ret/M_PI;

	return ret;
}

} // namespace

#endif
