/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>,
	Felix Lugauer, Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef NORMALIZED_MUTUAL_INFORMATION_H
#define NORMALIZED_MUTUAL_INFORMATION_H

#include "similarity_measure.h"

namespace vole {

/** 
* @class NormalizedMutualInformation 
* 
* @brief this class implements the computation of the normalized mutual information of two images
* 
* This metric is a quantity that measures the mutual dependence of the two random variables
* As we have images as inputs a probability distribution(histogram in this case) has to be calculated at first 
* Then this formula will be applied: NMI(A,B) = ( E(A) + E(B) ) / E(A,B)
* Where E(X) is the marginal entropy and E(X,Y) is the joint entropy of random variables X,Y
*
* This metric is optimal when its maximum is reached! 
* Its minimal value is 1 (one) because the metric is normalized!
*
* Note on the number of bins. 150 seemed to be a good value for our purpose and the tested images.
*
*/

template<typename T>
class NormalizedMutualInformation : public SimilarityMeasure<T> {

public:
	/**
	* constructor with histogram size
	*
	* @param bins number of histogram bins (which are uniformly distributed)
	* @param min lower inclusive bound of first histogram bin
	* @param max higher exclusive bound of last histogram bin
	*/
	NormalizedMutualInformation(int bins = 150, float min = 0.f, float max = 256.f)
	 : SimilarityMeasure<T>(), hbins(bins) { hrange[0] = min; hrange[1] = max; }
	
	/**
	* this function calculates the NMI metric by computing the histograms of both images and
	* the joinst historgran at first, afterwards these histograms will be used to calculate
	* the entropies in order to apply this formula: NMI(A,B) = ( E(A) + E(B) ) / E(A,B)
	*
	* @return floating-point number representing the measurement: maximal values are optimal
	*/
	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);

	int hbins;			//*< number of bins used for the histograms
	float hrange[2];	//*< value range of data
};

/* NOTE: code duplication except last line from mutual_information_histogram */
template<typename T>
inline double NormalizedMutualInformation<T>::getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
{
	this->check(img1, img2);
	double pixels = img1.rows * img1.cols;

	double entropy1 = 0., entropy2 = 0., entropy12 = 0.;

	// image histograms
	std::pair<cv::Mat_<float>, cv::Mat_<float> > hists =
		this->hist(img1, img2, hbins, hrange);
	// joint 2d histogram
	cv::Mat_<float> joint_pdf = this->jointhist(img1, img2, hbins, hrange);

	// calculate single image entropies
	for (int i = 0; i < hbins; i++) {

		double p1 = hists.first(i) / pixels; 
		if (p1 != 0)
			entropy1 += (-p1) * (log(p1) / log(2.0));

		double p2 = hists.second(i) / pixels;
		if (p2 != 0)
			entropy2 += (-p2) * (log(p2) / log(2.0));
	}

	// calculate entropy of the joint histogram
	for (int y = 0; y < hbins; y++) {
		for(int x = 0; x < hbins; x++) {
			 double p = joint_pdf(y,x) / pixels;
			 if (p != 0)
			 	entropy12 += (-p) * (log(p) / log(2.0));
		}
	}
	
	// NMI(A,B) = (E(A) + E(B)) / E(A,B)
	return (entropy1 + entropy2) / entropy12;
}

} // namespace

#endif // VOLE_NORMALIZED_MUTUAL_INFORMATION_H
