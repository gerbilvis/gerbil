/*
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>,
	David Foehrweiser.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef NORMALIZED_L2_H
#define NORMALIZED_L2_H

#include "similarity_measure.h"
#include <iostream>
#include <limits>

namespace vole {

/**
* @class Normalized Euclidean Distance by S. A. Robila 2005
*
*/
template<typename T>
class NormalizedL2 : public SimilarityMeasure<T> {

public:

	NormalizedL2() {}

	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);
};

template<typename T>
inline double NormalizedL2<T>::getSimilarity(const cv::Mat_<T> &v1, const cv::Mat_<T> &v2)
{
	this->check(v1, v2);

	cv::Scalar s1 = cv::mean(v1);
	cv::Scalar s2 = cv::mean(v2);
	if (s1[0] == 0. || s2[0] == 0.)
		return 0.;
		// can be harmful to graphseg: return (s1[0] == s2[0] ? 0. : std::numeric_limits<double>::max());

	v1 /= s1[0];
	v2 /= s2[0];

	return cv::norm(v1, v2, cv::NORM_L2);
}

} // namespace

#endif
