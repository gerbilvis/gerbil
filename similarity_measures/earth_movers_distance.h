/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>,
	Felix Lugauer, Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_EARTH_MOVERS_DISTANCE_H
#define VOLE_EARTH_MOVERS_DISTANCE_H

#include "similarity_measure.h"

namespace vole {

template<typename T>
class EarthMoversDistance : public SimilarityMeasure<T> {

public:

	EarthMoversDistance(int bins = 180, float min = 0.f, float max = 256.f)
		: SimilarityMeasure<T>(), hbins(bins) { hrange[0] = min; hrange[1] = max; }

	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);

	int hbins;			//*< number of bins used for the histograms
	float hrange[2];	//*< value range of data

};

template<typename T>
inline double EarthMoversDistance<T>::getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
{
	this->check(img1, img2);

	// image histograms
	std::pair<cv::Mat_<float>, cv::Mat_<float> > hists =
		this->hist(img1, img2, hbins, hrange);

	cv::Mat_<float> sig1(hbins, 2, 0.f);
	cv::Mat_<float> sig2(hbins, 2, 0.f);

	for (int i = 0; i < hbins; i++) {
		// TODO: use opencv row copying instead
		sig1(i, 0) = hists.first(i);
		sig1(i, 1) = (float)i;

		sig2(i, 0) = hists.second(i);
		sig2(i, 1) = (float)i;
	}

	return cv::EMD(sig1, sig2, CV_DIST_L2);
}

} // namespace

#endif // VOLE_EARTH_MOVERS_DISTANCE_H
