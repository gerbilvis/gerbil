/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>,
	Felix Lugauer, Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef SIMILARITY_MEASURE_H
#define SIMILARITY_MEASURE_H

#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <cassert>

/** 
* @mainpage Similarity Measures
* 
* Collection of distance measures for two grayscale images.
* So far it contains the following metrics:  
* - normalized mutual information
* - normalized cross correlation
* - mutual information (histogram)
* - mean squares
* - mean squares histogram
* - mean reciprocal square difference
* - correlation coefficient histogram
* - earth movers distance
* - gradient difference		<- computationally slow and under construction! 
*
*/

namespace vole {

/** 
* @class SimilarityMeasure 
* 
* @brief abstract class providing basic functions used for the similarity-calculation of two images
* 
* SimilarityMeasure is an abstract class that provides basic functions and data-containers
* in order to calculate the similarity of two images.
* Purpose of the methods implemented in this class is initialization and preparation of the data.
* Similarity will only be calculated on the selected areas (ROI).
* Calculations are done on roi1 and roi2, defined on img1 and img2.
* General Note:
* image1, image2, roi1 and roi2 must be of same size and only one channel.
*/

template<typename T>
class SimilarityMeasure {

public:
	/** 
	* empty constructor
	*/
	SimilarityMeasure() {}

    virtual ~SimilarityMeasure() {}

	// function for distance calculation that _must_ be implemented
	virtual double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)=0;

	// function for distance calculation with std::vector input
	/* this function _may_ be implemented for increased efficiency.
	   Default version builds CV matrix headers around the vector in O(1) and
	   then calls the other getSimilarity(). In cases where opencv is not needed,
	   e.g. simple distances, it is advised to overwrite this function. */
	virtual double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2);

	/* version of the method for implementing a position-based caching */
	virtual double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2,
	                             const cv::Point& c1, const cv::Point& c2);

	// helper function to check image input
	static void check(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
	{
		assert(!img1.empty() && !img2.empty());
		assert(img1.rows == img2.rows && img1.cols == img2.cols);
	}
	// helper function to check vector input
	static void check(const std::vector<T> &v1, const std::vector<T> &v2)
	{
		assert(!v1.empty());
		assert(v1.size() == v2.size());
	}

	// helper function to calculate histograms
	static std::pair<cv::Mat_<float>, cv::Mat_<float> >
	hist(const cv::Mat_<T> &in1, const cv::Mat_<T> &in2, int bins, float *range);
	// helper function to calculate joint histogram
	static cv::Mat_<float>
	jointhist(const cv::Mat_<T> &in1, const cv::Mat_<T> &in2, int bins, float *range);
};

template<typename T>
inline double SimilarityMeasure<T>::getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2)
{
	cv::Mat_<T> m1(v1), m2(v2);
	return getSimilarity(m1, m2);
}

template<typename T>
inline double SimilarityMeasure<T>::getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2,
                                                  const cv::Point &p1, const cv::Point &p2)
{
	return getSimilarity(v1, v2);
}

template<typename T>
std::pair<cv::Mat_<float>, cv::Mat_<float> >
SimilarityMeasure<T>::hist(const cv::Mat_<T> &in1, const cv::Mat_<T> &in2, int bins, float *range)
{
	cv::Mat_<float> hist1;	//*< histogram of the first img
	cv::Mat_<float> hist2;	//*< histogram of the second img

	// each has only one element (one-dimensional histogram)
	int histSize[] = { bins };
	const float* ranges[] = { range };
	int channels[] = { 0 }; // we compute the histogram for the 0-th channel

	// calc histogram of first input
	cv::calcHist(&in1, 1, channels, cv::Mat(), hist1, 1, histSize, ranges);

	// calc histogram of second input
	cv::calcHist(&in2, 1, channels, cv::Mat(), hist2, 1, histSize, ranges);
	
	return std::make_pair(hist1, hist2);
}

template<typename T>
cv::Mat_<float>
SimilarityMeasure<T>::jointhist(const cv::Mat_<T> &in1, const cv::Mat_<T> &in2, int bins, float *range)
{
	cv::Mat_<float> joint_pdf(bins, bins, 0.f);
	double ratio = bins / (range[1] - range[0]);

	for (int y = 0; y < in1.rows; y++) {
		const T *y1 = in1[y], *y2 = in2[y];
		for(int x = 0; x < in1.cols; x++) {
			int row = (int)floor((y1[x] - range[0]) * ratio);
			int col = (int)floor((y2[x] - range[0]) * ratio);
			joint_pdf(row, col) += 1;
		}
	}
	return joint_pdf;
}

} // namespace
#endif // SIMILARITY_MEASURE_H
