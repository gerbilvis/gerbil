/*
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>,
	David Foehrweiser.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_INF_DIV_H
#define VOLE_INF_DIV_H

#include "similarity_measure.h"
#include <iostream>
#include <algorithm>
#include <limits>

namespace similarity_measures {

/**
* @class Spectral Information Divergence
*
* @brief provides Kullback-Leibler-Style Distance Measurement
*
*/
template<typename T>
class SpectralInformationDivergence : public SimilarityMeasure<T> {

public:

	SpectralInformationDivergence() {}

	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);
};

template<typename T>
inline double SpectralInformationDivergence<T>::getSimilarity(const cv::Mat_<T> &p1, const cv::Mat_<T> &p2)
{
	this->check(p1, p2);

	cv::Scalar s1 = cv::sum(p1);
	cv::Scalar s2 = cv::sum(p2);
	if (s1[0] == 0. || s2[0] == 0.)
		return 0.;
		// can be harmful to graphseg: return (s1[0] == s2[0] ? 0. : std::numeric_limits<double>::max());

	p1 /= s1[0];
	p2 /= s2[0];

	cv::Mat_<T> l1, l2;
	cv::log(p1 / p2, l1);
	cv::log(p2 / p1, l2);
	cv::Scalar ret = cv::sum(p1.mul(l1)) + cv::sum(p2.mul(l2));
	return std::max(ret[0], 0.); // negative values come from strange pixels.
}

	/** The following code implements SID as it is defined in
		Spectral Matching Accuracy in Processing Hyperspectral Data
		Stefan A. Robila, 2005

	cv::Scalar mean1 = cv::mean(v1);
	cv::Scalar mean2 = cv::mean(v2);

	for (it1 = v1.begin(), it2 = v2.begin(); it1 < v1.end(); ++it1, ++it2) {

		if (mean1[0] <= 0) mean1[0] = 1.0f;
		if (mean2[0] <= 0) mean2[0] = 1.0f;
		double x = *it1 / mean1[0];
		double y = *it2 / mean2[0];

		double tmp1 = x - y;
		double tmp2 = std::log(x) - std::log(y);

		ret += tmp1*tmp2;
	}

	return std::abs<double>(ret);
	**/

} // namespace

#endif
