/*
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef SOM_DISTANCE
#define SOM_DISTANCE

#include "som.h"
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
	SOMDistance(SOM* som, int img_height, int img_width) : som(som),
		cache(som->createCache(img_height, img_width))
	{}
	~SOMDistance() { delete cache; }

	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);
	double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2);
	double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2,
						 const cv::Point &c1, const cv::Point &c2);

	SOM *const som;
	SOM::Cache *const cache;
};

template<typename T>
inline double SOMDistance<T>::getSimilarity(const cv::Mat_<T> &img1,
											const cv::Mat_<T> &img2)
{
	assert("use std::vector, this here would involve a copy" == 0);
	return getSimilarity(std::vector<T>(img1), std::vector<T>(img2));
}

template<typename T>
inline double SOMDistance<T>::getSimilarity(const std::vector<T> &v1,
											const std::vector<T> &v2)
{
	return som->getDistanceBetweenWinners(v1, v2);
}

template<typename T>
inline double SOMDistance<T>::getSimilarity(const std::vector<T> &v1,
											const std::vector<T> &v2,
											const cv::Point &c1,
											const cv::Point &c2)
{
	return cache->getDistance(v1, v2, c1, c2);
}

#endif
