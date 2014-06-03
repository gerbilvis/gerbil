/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>,
	Felix Lugauer, Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_MEAN_SQUARES_HISTOGRAM_H
#define VOLE_MEAN_SQUARES_HISTOGRAM_H

#include "similarity_measure.h"

namespace similarity_measures {

/** 
* @class MeanSquaresHistogram
* 
* @brief computes the entropy-based mean squared difference between two images
* 
* This metric is an alternative implementation of the mean squard metric using entropy.
* At first the histograms of both images and the joint histogram will be computed. 
* To these histograms this formula is applied: 1/N * SUM ( H12[x][y] * (H1(x) - H2(y))^2 )
* So we compute the mean squared difference of the histograms and 
* weight the difference with the pixelfrequency of the joint histogram
*
* This metric is optimal when its minimum is reached: 0 (zero)! 
*
* Note on the number of bins: a maximal value(256) seemed to be optimal for our purpose and the tested images.
*/
template<typename T>
class MeanSquaresHistogram : public SimilarityMeasure<T> {

public:
	/**
	* constructor with histogram size
	*
	* @param bins number of histogram bins (which are uniformly distributed)
	* @param min lower inclusive bound of first histogram bin
	* @param max higher exclusive bound of last histogram bin
	*/
	MeanSquaresHistogram(int bins = 150, float min = 0.f, float max = 256.f)
		: SimilarityMeasure<T>(), hbins(bins) { hrange[0] = min; hrange[1] = max; }

	/**
	* this function calculates the MSH metric by computing the histograms of both images and
	* the joinst historgran at first, afterwards these histograms will be used to apply 
	* this formula: 1/N * SUM ( H12[x][y] * (H1(x) - H2(y))^2 )
	*
	* @return floating-point number representing the measurement: minimal values(0) are optimal
	*/
	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);

	int hbins;			//*< number of bins used for the histograms
	float hrange[2];	//*< value range of data
};

template<typename T>
inline double MeanSquaresHistogram<T>::getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
{
	this->check(img1, img2);

	// image histograms
	std::pair<cv::Mat_<float>, cv::Mat_<float> > hists =
		this->hist(img1, img2, hbins, hrange);
	// joint 2d histogram
	cv::Mat_<float> joint_pdf = this->jointhist(img1, img2, hbins, hrange);

	double ret = 0., n = 0.;

	for (int y = 0; y < joint_pdf.rows; y++) {
		for (int x = 0; x < joint_pdf.cols; x++) {
			// weight the squared difference of hist1[y] and hist2[x] with joint[y][x] and sum it up
			float joint = joint_pdf(y, x);
			float diff = hists.first(y) - hists.second(x);
			ret += joint * diff*diff;
			n += joint;
			/* note: there was a bug here previously, with x and y mixed up. see
			   SimilarityMeasure::jointhist() for more indexing clarity */
		}
	}
	// average over the number of pixels in both images
	ret /= n;
	return ret;
}

} // namespace

#endif // VOLE_MEAN_SQUARES_HISTOGRAM_H
