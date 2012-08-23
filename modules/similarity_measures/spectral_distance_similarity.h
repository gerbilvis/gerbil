/*	
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>,
	David Foehrweiser.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_SPEC_DIST_SIM_H
#define VOLE_SPEC_DIST_SIM_H

#include "similarity_measure.h"
#include <math.h>

namespace vole {

template<typename T>
class SpectralDistanceSimilarity : public SimilarityMeasure<T> {

public:

	/**
	  @arg normType supported types are cv::NORM_L1, cv::NORM_L2, cv::NORM_INF
	  */
	SpectralDistanceSimilarity(int normType = cv::NORM_L2) : normType(normType), maxVal(1.0f), minVal(0.0f) {}
	SpectralDistanceSimilarity(double maxVal, double minVal, int normType = cv::NORM_L2) : normType(normType), maxVal(maxVal), minVal(minVal) {}

	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);
	double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2);

	int normType;
	double maxVal;
	double minVal;
};

template<typename T>
inline double SpectralDistanceSimilarity<T>::getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
{
	check(img1, img2);

	cv::MatConstIterator_<T> it1 = img1.begin(), it2 = img2.begin();
	double normVal = 0.0f;

	normVal = cv::norm(img1, img2, normType);
	
	if ((maxVal-minVal) == 0) return normVal;

	normVal = (normVal - minVal)/(maxVal - minVal);

	return normVal;

}

template<typename T>
inline double SpectralDistanceSimilarity<T>::getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2)
{
	check(v1, v2);

	typename std::vector<T>::const_iterator it1 = v1.begin(), it2 = v2.begin();

	double normVal = 0.0f;

	normVal = cv::norm(cv::Mat_<T>(v1), cv::Mat_<T>(v2), normType);
	
	if ((maxVal-minVal) == 0) return normVal;

	normVal = (normVal - minVal)/(maxVal - minVal);
	return normVal;
}

} // namespace

#endif
