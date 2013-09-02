/*	
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>,
	David Foehrweiser.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_CORR_SIM_H
#define VOLE_CORR_SIM_H

#include "similarity_measure.h"
#include <iostream>
#include <math.h>

#ifndef M_PI
#define M_PI	3.14159265358979323846264338327950288419716939937f
#endif

namespace vole {

/**
* @class SpectralCorrelationSimilarity
*
* @brief provides Pearson Correlation normalized between [0,1]
*
*/
template<typename T>
class SpectralCorrelationSimilarity : public SimilarityMeasure<T> {

public:

	SpectralCorrelationSimilarity() {}

	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);
	double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2);
};

template<typename T>
inline double SpectralCorrelationSimilarity<T>::getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
{
	check(img1, img2);
	
	cv::MatConstIterator_<T> it1 = img1.begin(), it2 = img2.begin();
	int n = img1.size().height;
		
	double ret = 0.0f;
	cv::Scalar mean1, mean2;
	cv::Scalar stdDev1, stdDev2;

	cv::meanStdDev(img1, mean1, stdDev1);
	cv::meanStdDev(img2, mean2, stdDev2);

	for (; it1 != img1.end(); ++it1, ++it2) {

		double tmp1 = *it1 - mean1[0];
		double tmp2 = *it2 - mean2[0];

		ret += tmp1 * tmp2;
	}

	
	ret = ret/((n-1.0f)*(stdDev1[0]*stdDev2[0]));

	return ret;

}

template<typename T>
inline double SpectralCorrelationSimilarity<T>::getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2)
{
	check(v1, v2);

	typename std::vector<T>::const_iterator it1 = v1.begin(), it2 = v2.begin();
	double ret = 0.0f;
	float n = static_cast<float>(v1.size());

	/* compute mean */
	cv::Scalar mean1, mean2;
	cv::Scalar stdDev1, stdDev2;

	cv::meanStdDev(cv::Mat_<T>(v1), mean1, stdDev1);
	cv::meanStdDev(cv::Mat_<T>(v2), mean2, stdDev2);

	/*compute standard deviation */
	for (it1 = v1.begin(), it2 = v2.begin(); it1 < v1.end(); ++it1, ++it2) {

		float tmp1 = *it1 - mean1[0];
		float tmp2 = *it2 - mean2[0];

		ret += static_cast<double>(tmp1 * tmp2);
	}


	/*return spectral correlation */
	ret = ret/((n)*(stdDev1[0]*stdDev2[0]));
	
	return ret;
}


} // namespace

#endif
