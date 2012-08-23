/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>,
	Felix Lugauer, Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_MEAN_RECIPROCAL_SQUARE_DIFFERENCE_H
#define VOLE_MEAN_RECIPROCAL_SQUARE_DIFFERENCE_H

#include "similarity_measure.h"

namespace vole {

/** 
* @class MeanReciprocalSquareDifference
*
* @brief implements the calculation of the MeanReciprocalSquareDifference metric
* 
* MRSD computes the pixel-wise differences and 
* sums them up after passing them through a bell-shaped function.
* Formula: SUM( 1 / (1 + (A[i] - B[i])^2/lambda^2 ) )
*
* This metric is optimal when its maximum (= Number of pixels considered) is reached!
*
* Note on parameter lambda: important value that controls the capture radius!
* Tests showed that a bigger capture radius[20,40] gained more notable results so far!
*
* Further description from orfeo-toolbox.org:
* "This image metric has the advantage of producing poor values 
* when few pixels are considered. This makes it consistent when its computation
* is subject to the size of the overlap region between the images. 
* The capture radius of the metric can be regulated with the parameter lambda. 
* The profile of this metric is very peaky. The sharp peaks of the metric 
* help to measure spatial misalignment with high precision. 
* Note that the notion of capture radius is used here in terms of the intensity domain,
* not the spatial domain. In that regard, lambda should be given in intensity units
* and be associated with the differences in intensity that will make drop the metric by 50%"
*/
template<typename T>
class MeanReciprocalSquareDifference : public SimilarityMeasure<T> {

public:
	/**
	* constructor sets the capture radius lambda
	*
	* @param lambda capture radius (20 <= lambda <= 40 looks reasonable)
	*/
	MeanReciprocalSquareDifference(double lambda = 30)
		: SimilarityMeasure<T>(), lambda(lambda) {}

	/**
	* this function calculates the MRSD metric by computing the pixel-wise differences and 
	* by summing them up after passing them through a bell-shaped function:
	* Formula: SUM( 1 / (1 + (A[i] - B[i])^2/lambda^2 ) )
	*
	* @return floating-point number representing the measurement: maximal values are optimal
	*/
	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);
	double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2);

	double lambda;	//*< capture radius
};

template<typename T>
inline double MeanReciprocalSquareDifference<T>::getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
{
	this->check(img1, img2);

	double ret = 0.;
	cv::MatConstIterator_<T> it1 = img1.begin(), it2 = img2.begin();
	for (; it1 < img1.end(); ++it1, ++it2) {

		// compute the pixel-wise difference
		double diff = *it1 - *it2;

		// sum up differences after passing diff throug a bell shaped function
		ret += 1.0f / (1.0f + ((diff*diff) / (lambda*lambda)));
	}
	return ret;
}

template<typename T>
inline double MeanReciprocalSquareDifference<T>::getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2)
{
	this->check(v1, v2);

	double ret = 0.;
	typename std::vector<T>::const_iterator it1 = v1.begin(), it2 = v2.begin();
	for (; it1 < v1.end(); ++it1, ++it2) {

		// compute the pixel-wise difference
		double diff = *it1 - *it2;

		// sum up differences after passing diff throug a bell shaped function
		ret += 1.0f / (1.0f + ((diff*diff) / (lambda*lambda)));
	}
	return ret;
}

} // namespace

#endif // VOLE_MEAN_RECIPROCAL_SQUARE_DIFFERENCE_H
