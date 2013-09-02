/*
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>,
	David Foehrweiser.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef SIDSAM_H
#define SIDSAM_H

#include "modified_spectral_angle_similarity.h"
#include "spectral_information_divergence.h"
#include <iostream>
#include <limits>

namespace vole {

/**
* @class Measure based on Spectral Angle Similarity and
		 Spectral Information Divergence
*
*/
template<typename T>
class SIDSAM : public SimilarityMeasure<T> {

public:

	SIDSAM(int version) : v(version)
	{ assert(v == 0 || v == 1); }

	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);

	int v;
	ModifiedSpectralAngleSimilarity<T> sam;
	SpectralInformationDivergence<T> sid;
};

template<typename T>
inline double SIDSAM<T>::getSimilarity(const cv::Mat_<T> &v1, const cv::Mat_<T> &v2)
{
	if (v == 0) {
		return std::sqrt(sid.getSimilarity(v1, v2) * std::sin(sam.getSimilarity(v1, v2)));
	} else {
		return std::sqrt(sid.getSimilarity(v1, v2) * std::tan(sam.getSimilarity(v1, v2)));
	}
}

} // namespace

#endif
