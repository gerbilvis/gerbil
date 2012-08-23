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
	  @arg h height of the pixel cache (image size, not som size)
	  @arg w width of the pixel cache
	  */
	SOMDistance(const SOM* som, int h, int w) : som(som), hack3d(som->ishack3d()),
		cache(std::vector<std::vector<cv::Point> >(h,
			  std::vector<cv::Point>(w, cv::Point(-1, -1))))
	{}

	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);
	double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2);
	double getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2,
						 const cv::Point& c1, const cv::Point& c2);

	const SOM *som;
	const bool hack3d;
	std::vector<std::vector<cv::Point> > cache;
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

template<typename T>
inline double SOMDistance<T>::getSimilarity(const std::vector<T> &v1, const std::vector<T> &v2,
											const cv::Point& c1, const cv::Point& c2)
{
	cv::Point &p1 = cache[c1.y][c1.x];
	cv::Point &p2 = cache[c2.y][c2.x];
	if (p1 == cv::Point(-1, -1))
		p1 = som->identifyWinnerNeuron(v1);

	if (p2 == cv::Point(-1, -1))
		p2 = som->identifyWinnerNeuron(v2);

	if (hack3d) {
		cv::Point3d p1_3(p1.x, p1.y / som->getWidth(), p1.y % som->getWidth());
		cv::Point3d p2_3(p2.x, p2.y / som->getWidth(), p2.y % som->getWidth());
		return som->getDistance3(p1_3, p2_3);
	} else {
		return som->getDistance(p1, p2);
	}
}

#endif
