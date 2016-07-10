/*
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>,
	Felix Lugauer, Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_L_NORM_H
#define VOLE_L_NORM_H

#include "similarity_measure.h"
#include <xmmintrin.h>
#include <emmintrin.h>
#include <algorithm>

namespace similarity_measures {

/**
* @class LNorm
*
* @brief provides L*-norm distance (Manhattan, Euclidean, Chebyshev)
*
*/
template<typename T>
class LNorm : public SimilarityMeasure<T> {

public:

	/**
	  @arg normType supported types are cv::NORM_L1, cv::NORM_L2, cv::NORM_INF
	  */
	LNorm(int normType = cv::NORM_L2) : normType(normType) {}

	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);
	double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2);

	int normType;
};

template<typename T>
inline double LNorm<T>::getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
{
	this->check(img1, img2);

	return cv::norm(img1, img2, normType);
}

template<typename T>
inline double LNorm<T>::getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2)
{
	this->check(v1, v2);

	double ret = 0.;
	typename std::vector<T>::const_iterator it1 = v1.begin(), it2 = v2.begin();
	switch (normType) {
	case cv::NORM_L1:
		for (; it1 < v1.end(); ++it1, ++it2)
			ret += std::abs(*it1 - *it2);
		break;
	case cv::NORM_L2:
		for (; it1 < v1.end(); ++it1, ++it2) {
			double diff = *it1 - *it2;
			ret += diff * diff;
		}
		ret = std::sqrt(ret);
		break;
	case cv::NORM_INF:
		for (; it1 < v1.end(); ++it1, ++it2) {
			double diff = std::abs<T>(*it1 - *it2);
			ret = std::max<double>(diff, ret);
		}
		break;
	default:
		assert(normType != normType);
	}
	return ret;
}

template<>
inline double LNorm<float>::getSimilarity(const std::vector<float> &v1, const std::vector<float> &v2)
{
	this->check(v1, v2);

	double ret = 0.;
	std::vector<float>::const_iterator it1 = v1.begin(), it2 = v2.begin();
	switch (normType) {
	case cv::NORM_L1:
		for (; it1 < v1.end(); ++it1, ++it2)
			ret += std::abs(*it1 - *it2);
		break;
	case cv::NORM_L2:
	{
		int i = 0;
		__m128 vret = _mm_setzero_ps();
		for (; i < (int)v1.size() - 4; i += 4) {
// GNU malloc guarantees 16 bit alignment on 64 bit platforms
#if defined(__GNUC__) && defined(__LP64__)
			__m128 vec1 = _mm_load_ps(&v1[i]);
			__m128 vec2 = _mm_load_ps(&v2[i]);
#else
			__m128 vec1 = _mm_loadu_ps(&v1[i]);
			__m128 vec2 = _mm_loadu_ps(&v2[i]);
#endif
			__m128 vdiff = _mm_sub_ps(vec1, vec2);
			__m128 vdiff2 = _mm_mul_ps(vdiff, vdiff);
			vret = _mm_add_ps(vret, vdiff2);
		}
		ret += *((float*)&vret + 0);
		ret += *((float*)&vret + 1);
		ret += *((float*)&vret + 2);
		ret += *((float*)&vret + 3);
		for (; i < int(v1.size()); i++) {
			float diff = v1[i] - v2[i];
			ret += diff * diff;
		}
		ret = std::sqrt(ret);
		break;
	}
	case cv::NORM_INF:
		for (; it1 < v1.end(); ++it1, ++it2) {
			double diff = std::abs<float>(*it1 - *it2);
			ret = std::max<double>(diff, ret);
		}
		break;
	default:
		assert(normType != normType);
	}
	return ret;
}

template<>
inline double LNorm<double>::getSimilarity(const std::vector<double> &v1, const std::vector<double> &v2)
{
	this->check(v1, v2);

	double ret = 0.;
	std::vector<double>::const_iterator it1 = v1.begin(), it2 = v2.begin();
	switch (normType) {
	case cv::NORM_L1:
		for (; it1 < v1.end(); ++it1, ++it2)
			ret += std::abs(*it1 - *it2);
		break;
	case cv::NORM_L2:
	{
		const double* x1 = &v1[0];
		const double* x2 = &v2[0];
		int i = 0;
		__m128d vret = _mm_setzero_pd();
		for (; i < (int)v1.size() - 2; i += 2) {
#if defined(__GNUC__) && defined(__LP64__)
			__m128d vec1 = _mm_load_pd(x1);
			__m128d vec2 = _mm_load_pd(x2);
#else
			__m128d vec1 = _mm_loadu_pd(x1);
			__m128d vec2 = _mm_loadu_pd(x2);
#endif
			__m128d vdiff = _mm_sub_pd(vec1, vec2);
			__m128d vdiff2 = _mm_mul_pd(vdiff, vdiff);
			vret = _mm_add_pd(vret, vdiff2);
			x1 += 2;
			x2 += 2;
		}
		ret += *((double*)&vret + 0);
		ret += *((double*)&vret + 1);
		for (; i < int(v1.size()); i++) {
			double diff = *x1 - *x2;
			ret += diff * diff;
			x1++;
			x2++;
		}
		ret = std::sqrt(ret);
		break;
	}
	case cv::NORM_INF:
		for (; it1 < v1.end(); ++it1, ++it2) {
			double diff = std::abs<double>(*it1 - *it2);
			ret = std::max<double>(diff, ret);
		}
		break;
	default:
		assert(normType != normType);
	}
	return ret;
}

} // namespace

#endif
