/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>,
	Felix Lugauer, Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_MUTUAL_INFORMATION_HISTOGRAM_H
#define VOLE_MUTUAL_INFORMATION_HISTOGRAM_H

#include "similarity_measure.h"

namespace similarity_measures {

/** 
* @class MutualInformationHistogram
* 
* @brief this class implements the computation of the mutual information of two images
* 
* This metric measures the mutual dependence of the two random variables
* Mutual information (MI) measures how much information one random variable
* (image intensity in one image) tells about another random variable (image intensity in the other image) 
* This formula will be applied: MI(A,B) = E(A) + E(B) - E(A,B)
* Where E(X) is the marginal entropy and E(X,Y) is the joint entropy of random variables X,Y
*
* This metric is optimal when its maximum is reached! 
*
* Note on the number of bins. 150 seemed to be a good value for our purpose and the tested images.
*/
template<typename T>
class MutualInformationHistogram : public SimilarityMeasure<T> {

public:
	/**
	* constructor sets the histogram size
	*
	* @param bins number of bins for the histograms (important parameter)
	*/
	MutualInformationHistogram(int bins = 150, float min = 0.f, float max = 256.f)
		: SimilarityMeasure<T>(), hbins(bins) { hrange[0] = min; hrange[1] = max; }

	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);

	int hbins;			//*< number of bins used for the histograms
	float hrange[2];	//*< value range of data
};

template<typename T>
inline double MutualInformationHistogram<T>::getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
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

	// MI(A,B) = E(A) + E(B) - E(A,B)
	return entropy1 + entropy2 - entropy12;
}

} // namespace

#endif // VOLE_MUTUAL_INFORMATION_HISTOGRAM_H
