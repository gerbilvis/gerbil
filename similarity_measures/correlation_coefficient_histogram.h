/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>,
	Felix Lugauer, Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_CORRELATION_COEFFICIENT_HISTOGRAM_H
#define VOLE_CORRELATION_COEFFICIENT_HISTOGRAM_H

#include "similarity_measure.h"

namespace similarity_measures {

template<typename T>
class CorrelationCoefficientHistogram : public SimilarityMeasure<T> {

public:
	CorrelationCoefficientHistogram(int bins = 150, float min = 0.f, float max = 256.f)
		: SimilarityMeasure<T>(), hbins(bins) { hrange[0] = min; hrange[1] = max; }

	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);

	int hbins;			//*< number of bins used for the histograms
	float hrange[2];	//*< value range of data

};

template<typename T>
inline double CorrelationCoefficientHistogram<T>::getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
{
	this->check(img1, img2);

	// image histograms
	std::pair<cv::Mat_<float>, cv::Mat_<float> > hists =
		this->hist(img1, img2, hbins, hrange);

	// correlate histograms
	return cv::compareHist(hists.first, hists.second, CV_COMP_CORREL);
}

} // namespace

#endif // VOLE_CORRELATION_COEFFICIENT_HISTOGRAM_H
