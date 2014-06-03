/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>,
	Felix Lugauer, Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_NORMALIZED_CROSS_CORRELATION_H
#define VOLE_NORMALIZED_CROSS_CORRELATION_H

#include "similarity_measure.h"

namespace similarity_measures {

/** 
* @class NormalizedCrossCorrelation 
* 
* @brief this class implements the calculation of the normalized pixel-wise cross-correlation of two images
* 
* NormalizedCrossCorrelation computes pixel-wise cross-correlation and normalizes it by the square root
* of the autocorrelation of the images.
*
* This metric is optimal when its minimum is reached so its optimal value is -1 (minus one)!
*
* Further description from orfeo-toolbox.org:
* "Misalignment between the images results in small measure values. 
* The use of this metric is limited to images obtained using the same imaging modality. 
* The metric is insensitive to multiplicative factors, illumination changes, between the two images.
* This metric produces a cost function with sharp peaks and well defined minima. 
* On the other hand, it has a relatively small capture radius."
*/
template<typename T>
class NormalizedCrossCorrelation : public SimilarityMeasure<T> {

public:

	/**
	* this function computes the pixel-wise cross-correlation and normalizes it by the square root
	* of the autocorrelation of the images
	* @return floating-point number representing the measurement: -1 is the optimal value
	*/
	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);
	double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2);
};

template<typename T>
inline double NormalizedCrossCorrelation<T>::getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
{
	this->check(img1, img2);

	double corr1 = 0., corr2 = 0., crosscorr = 0.;
	
	cv::MatConstIterator_<T> it1 = img1.begin(), it2 = img2.begin();
	for (; it1 < img1.end(); ++it1, ++it2) {
		double v1 = *it1, v2 = *it2;
		// auto correlation of both images
		corr1 += v1*v1;
		corr2 += v2*v2;
		// pixel-wise cross correlation
		crosscorr += v1*v2;
	}
	// normalizing the pixel cross correlation by the square root of the autocorrelation of the images
	return (-crosscorr) / (sqrt(corr1) * sqrt(corr2));
}

template<typename T>
inline double NormalizedCrossCorrelation<T>::getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2)
{
	this->check(v1, v2);

	double corr1 = 0., corr2 = 0., crosscorr = 0.;

	typename std::vector<T>::const_iterator it1 = v1.begin(), it2 = v2.begin();
	for (; it1 < v1.end(); ++it1, ++it2) {
		double v1 = *it1, v2 = *it2;
		// auto correlation of both images
		corr1 += v1*v1;
		corr2 += v2*v2;
		// pixel-wise cross correlation
		crosscorr += v1*v2;
	}
	// normalizing the pixel cross correlation by the square root of the autocorrelation of the images
	return (-crosscorr) / (sqrt(corr1) * sqrt(corr2));
}

} // namespace

#endif // VOLE_NORMALIZED_CROSS_CORRELATION_H
