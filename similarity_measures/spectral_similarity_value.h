/*	
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>,
	David Foehrweiser.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_SPEC_SIM_VAL_H
#define VOLE_SPEC_SIM_VAL_H

#include "similarity_measure.h"
#include <spectral_correlation_similarity.h>
#include <spectral_distance_similarity.h>
#include <math.h>

#ifndef M_PI
#define M_PI	3.14159265358979323846264338327950288419716939937f
#endif

namespace vole {

/**
* @class Spectral Similarity Value
*
* @brief provides spectral similarity value distance
*
*/
template<typename T>
class SpectralSimilarityValue : public SimilarityMeasure<T> {

public:

	/**
	  @arg normType supported types are cv::NORM_L1, cv::NORM_L2, cv::NORM_INF
	  */
	SpectralSimilarityValue(int normType = cv::NORM_L2) : normType(normType), maxVal(1.0f), minVal(0.0f) {}
	SpectralSimilarityValue(double maxVal, double minVal, int normType = cv::NORM_L2) : normType(normType), maxVal(maxVal), minVal(minVal) {}


	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);
	double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2);

	int normType;
	double maxVal;
	double minVal;

};

template<typename T>
inline double SpectralSimilarityValue<T>::getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
{
	check(img1, img2);
	double ret;
	SpectralDistanceSimilarity<T> specDistFun(std::sqrt(maxVal*maxVal), std::sqrt(minVal*minVal), normType);
	SpectralCorrelationSimilarity<T> specCorrDistFun;

	double euclideanDist = specDistFun.getSimilarity(img1, img2);
	double spectralCorr = specCorrDistFun.getSimilarity(img1, img2);

	/*normalize spectralCorr to let it be between [0,1] it is now called spectral correlation angle*/
	spectralCorr = std::acos((spectralCorr + 1.0f)/2.0f);
	spectralCorr /= std::acos(0.0f);

	ret = (euclideanDist * euclideanDist) - ((1.0f-spectralCorr) * (1.0f-spectralCorr));
	if (ret < 0.0f)
		ret = (euclideanDist * euclideanDist);

	ret = std::sqrt(ret);
	return ret;
}

template<typename T>
inline double SpectralSimilarityValue<T>::getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2)
{
	check(v1, v2);
	double ret;
	SpectralDistanceSimilarity<T> specDistFun(std::sqrt(maxVal*maxVal), std::sqrt(minVal*minVal), normType);
	SpectralCorrelationSimilarity<T> specCorrDistFun;

	double euclideanDist = specDistFun.getSimilarity(v1, v2);
	double spectralCorr = specCorrDistFun.getSimilarity(v1, v2);
	
	/*normalize spectralCorr to let it be between [0,1] it is now called spectral correlation angle*/
	spectralCorr = std::acos((spectralCorr + 1.0f)/2.0f);
	spectralCorr /= std::acos(0.0f);

	ret = (euclideanDist * euclideanDist) - ((1.0f-spectralCorr) * (1.0f-spectralCorr));
	if (ret < 0.0f)
		ret = (euclideanDist * euclideanDist);

	ret = std::sqrt(ret);
	return ret;
}

} // namespace

#endif
