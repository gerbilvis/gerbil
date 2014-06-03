/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>,
	Felix Lugauer, Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_MEAN_SQUARES_H
#define VOLE_MEAN_SQUARES_H

#include "similarity_measure.h"

namespace similarity_measures {

/** 
* @class MeanSquares
* 
* @brief provides the computation of the mean squared pixel-wise intensity difference between two images
* 
* This metric calculates the mean squared pixel-wise difference between image A and B.
* As formula: MS(A,B) = 1 / N * SUM( (A[i] - B[i])^2 ) 
*
* This metric is optimal when its minium is reached: 0 (zero)! 
* No additional parameters are needed for the calculation.
*
* Further description from orfeo-toolbox.org:
* "This metric is simple to compute and has a relatively large capture radius. 
* This metric relies on the assumption that intensity representing the same homologous point
* must be the same in both images. Hence, its use is restricted to images of the same modality. 
* Additionally, any linear changes in the intensity result in a poor match value."
* 
*/
template<typename T>
class MeanSquares : public SimilarityMeasure<T> {

public:
	
	/**
	* calculates the MeanSquares metric by summing up the pixel-wise squared 
	* difference of the images and averages the result
	*/
	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);
	double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2);
};

template<typename T>
inline double MeanSquares<T>::getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
{
	this->check(img1, img2);
		
	double ret = 0.;
	cv::MatConstIterator_<T> it1 = img1.begin(), it2 = img2.begin();
	for (; it1 < img1.end(); ++it1, ++it2) {
		double diff = *it1 - *it2;
		ret += (diff * diff);
	}
	// average the result
	ret /= img1.rows*img1.cols;
	
	return ret;
}

template<typename T>
inline double MeanSquares<T>::getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2)
{
	this->check(v1, v2);

	double ret = 0.;
	typename std::vector<T>::const_iterator it1 = v1.begin(), it2 = v2.begin();
	for (; it1 < v1.end(); ++it1, ++it2) {
		double diff = *it1 - *it2;
		ret += (diff * diff);
	}
	return ret;
}

} // namespace

#endif // VOLE_MEAN_SQUARES_H
