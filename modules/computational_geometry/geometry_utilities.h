#ifndef VOLE_GEOMETRY_UTILITIES_H_
#define VOLE_GEOMETRY_UTILITIES_H_


//#include "ContainerTraits.h"

// #include "data_traits.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <iostream>
#include <ostream>
#include <algorithm>

namespace vole {

class GeometryUtilities
{
public:

	static double euclid_dist(cv::Vec2f &a, cv::Vec2f &b);
	static double euclid_dist_square(cv::Vec2f &a, cv::Vec2f &b);

	/** computes the intersection of a line (point + direction) with its perpendicular line that goes through point2

		\param Startpunkt
		\param Richtungsvektor, der an den Startpunkt gehaengt wird.
		\param Punkt, von dem das Lot auf die durch p1 und dir beschriebene Gerade gefaellt werden soll.
		\return Schnittpunkt.
	*/
	static cv::Vec2f raisePerpendicular(const cv::Vec2f p1, const cv::Vec2f dir, const cv::Vec2f perp);

	/** this variant of raisePerpendicular returns the result in a Point2D<T> reference "result"
	*/
	static void raisePerpendicular(const cv::Vec2f p1, const cv::Vec2f dir, const cv::Vec2f perp, cv::Vec2f &result);

	static double distSquare(const cv::Vec2f p1, const cv::Vec2f p2);

	static bool isClockwise(const cv::Vec2f p1, const cv::Vec2f p2, const cv::Vec2f p3);

	static bool isCounterClockwise(const cv::Vec2f p1, const cv::Vec2f p2, const cv::Vec2f p3);
	static double scalarProd(cv::Vec2f v1, cv::Vec2f v2);

	static bool spansPlane(const cv::Vec3d &x, const cv::Vec3d &y, const cv::Vec3d &z);
	static bool spansPlane(const std::vector<cv::Vec3d> &points);

	
	/** clockwise gives an indicator for the relative position of 3 points:
	- the points are colinear if clockwise == 0
	- the points are in clockwise order (= describe a turn right) if clockwise > 0
	- the points are in counterclockwise order if clockwise < 0
	this is an extremely cheap operation that can be used as a sorting criterion for the graham scan algorithm
	*/
	static int clockwise(const cv::Vec2f idx1, const cv::Vec2f idx2, const cv::Vec2f idx3);

	/** tests a point for being on an edge delimited by two points
	 */
	static bool pointOnEdge(const cv::Vec2f point, const cv::Vec2f edge_point1, const cv::Vec2f edge_point2);

	/** tests a point for being inside (or on the boundary of) a convex (!) polygon
	 *  \param the test point
	 *  \param the polygon, represented as a ordered sequence of edge points
	 *  \return true if the point is in the polygon or on its boundary
	 */
	static bool pointInPolygon(const cv::Vec2f point, const std::vector<cv::Vec2f> &polygon);
};

}

#endif // VOLE_GEOMETRY_UTILITIES_H_

