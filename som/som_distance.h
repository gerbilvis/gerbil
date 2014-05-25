/*
	Copyright(c) 2014 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef SOM_DISTANCE
#define SOM_DISTANCE

#include "gensom.h"
#include "som_cache.h"
#include <l_norm.h>
#include <similarity_measure.h>

/**
* @class SOMDistance
*
* @brief provides SOM distance as a similarity measure
*
*/
template<typename T>
class SOMDistance : public vole::SimilarityMeasure<T> {

public:

	/**
	  @arg som Already trained Self-Organizing map
	  @arg img_height height of the pixel cache (image size, not som size)
	  @arg img_width width of the pixel cache
	  */
	SOMDistance(const GenSOM &som, const multi_img& img)
		: som(som), img(img), cache(SOMClosestN(som, img, 1))
	{}

	double getSimilarity(const cv::Mat_<T> &m1, const cv::Mat_<T> &m2);
	double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2);
	double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2,
						 const cv::Point &c1, const cv::Point &c2);

	const GenSOM &som;
	const multi_img &img;
	SOMClosestN cache;
	vole::LNorm<T> l2;
};

template<typename T>
inline double SOMDistance<T>::getSimilarity(const cv::Mat_<T> &m1,
											const cv::Mat_<T> &m2)
{
	assert("use std::vector, this here would involve a copy" == 0);
	return getSimilarity(std::vector<T>(m1), std::vector<T>(m2));
}

template<typename T>
inline double SOMDistance<T>::getSimilarity(const std::vector<T> &v1,
											const std::vector<T> &v2)
{
	std::vector<float> n1 = som.getCoord(som.findBMU(v1).index);
	std::vector<float> n2 = som.getCoord(som.findBMU(v2).index);
	return l2.getSimilarity(n1, n2);
}

template<typename T>
inline double SOMDistance<T>::getSimilarity(const std::vector<T> &v1,
											const std::vector<T> &v2,
											const cv::Point &c1,
											const cv::Point &c2)
{
	std::vector<float> n1 = som.getCoord(cache.closestN(c1).first->index);
	std::vector<float> n2 = som.getCoord(cache.closestN(c2).first->index);
	return l2.getSimilarity(n1, n2);
}

#endif
