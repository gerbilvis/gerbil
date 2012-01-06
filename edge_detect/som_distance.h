/*
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef SOM_DISTANCE
#define SOM_DISTANCE

#include "self_organizing_map.h"
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
	  */
	SOMDistance(const SOM* som) : som(som), hack3d(som->ishack3d()) {}

	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);
	double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2);

	const SOM *som;
	const bool hack3d;
};

template<typename T>
inline double SOMDistance<T>::getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
{
	assert("use std::vector, this here would involve a copy" == 0);
	return getSimilarity(std::vector<T>(img1), std::vector<T>(img2));
}

template<typename T>
inline double SOMDistance<T>::getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2)
{
	cv::Point p1 = som->identifyWinnerNeuron(v1);
	cv::Point p2 = som->identifyWinnerNeuron(v2);

	if (hack3d) {
		cv::Point3d p1_3(p1.x, p1.y / som->getWidth(), p1.y % som->getWidth());
		cv::Point3d p2_3(p2.x, p2.y / som->getWidth(), p2.y % som->getWidth());
		return som->getDistance3(p1_3, p2_3);
	} else {
		return som->getDistance(p1, p2);
	}
}

#endif
