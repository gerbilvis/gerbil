#include "geometry_utilities.h"


namespace vole {


double GeometryUtilities::euclid_dist(cv::Vec2f &a, cv::Vec2f &b)
{
	float t0 = a[0] - b[0];
	float t1 = a[1] - b[1];
	return std::sqrt( t0*t0 + t1*t1 );
}

double GeometryUtilities::euclid_dist_square(cv::Vec2f &a, cv::Vec2f &b)
{
	float t0 = a[0] - b[0];
	float t1 = a[1] - b[1];
	return t0*t0 + t1*t1;
}


/** computes the intersection of a line (point + direction) with its perpendicular line that goes through point2
	\param Startpunkt
	\param Richtungsvektor, der an den Startpunkt gehaengt wird.
	\param Punkt, von dem das Lot auf die durch p1 und dir beschriebene Gerade gefaellt werden soll.
	\return Schnittpunkt.
*/
cv::Vec2f GeometryUtilities::raisePerpendicular(const cv::Vec2f p1, const cv::Vec2f dir, const cv::Vec2f perp)
{
	double enumerator = p1[1]*dir[0] + perp[0] * dir[1] - p1[0]*dir[1] - dir[0]*perp[1];
	double denominator = dir[0] * dir[0] + dir[1] * dir[1];
	double h = (double) enumerator / (double)denominator;
	return cv::Vec2f(static_cast<float>(perp[0] - h*dir[1]), static_cast<float>(perp[1] + h*dir[0]));
}

/** this variant of raisePerpendicular returns the result in a Point2D<T> reference "result"
*/
void GeometryUtilities::raisePerpendicular(const cv::Vec2f p1, const cv::Vec2f dir, const cv::Vec2f perp, cv::Vec2f &result)
{
	double enumerator = dir[0]*(p1[1] - perp[1]) + dir[1]*(perp[0] - p1[0]);
	double denominator = dir[0] * dir[0] + dir[1] * dir[1];
	double h = (double) enumerator / (double)denominator;
	result[0] = static_cast<float>(perp[0] - h*dir[1]);
	result[1] = static_cast<float>(perp[1] + h*dir[0]);

	float test0 = static_cast<float>(perp[0] + h*dir[1]);
	float test1 = static_cast<float>(perp[1] - h*dir[0]);

	float dbl_check1 = (result[0] - p1[0])*(result[0] - p1[0]) + (result[1] - p1[1])*(result[1] - p1[1]);
	float dbl_check2 = (test0 - p1[0])*(test0 - p1[0]) + (test1 - p1[1])*(test1 - p1[1]);
	if (dbl_check2 < dbl_check1) {
		std::cout << "oops, dbl_check2 (" << dbl_check2 << ") < dbl_check1 (" << dbl_check1 << ")" << std::endl;
//	} else {
//		std::cout << "ok..." << std::endl;
	}
}

double GeometryUtilities::distSquare(const cv::Vec2f p1, const cv::Vec2f p2)
{
	return (p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1]);
}

bool GeometryUtilities::isClockwise(const cv::Vec2f p1, const cv::Vec2f p2, const cv::Vec2f p3)
{
	return clockwise(p1, p2, p3) < 0;
}

bool GeometryUtilities::spansPlane(const cv::Vec3d &x, const cv::Vec3d &y, const cv::Vec3d &z)
{
	// compute direction vectors y-x, z-x
	cv::Vec3d d1(y[0] - x[0], y[1] - x[1], y[2] - x[2]);
	cv::Vec3d d2(z[0] - x[0], z[1] - x[1], z[2] - x[2]);
	
	// cross-product between these vectors is zero length? points are colinear -> no plane
	cv::Vec3d cross_prod(d1[1]*d2[2] - d2[1]*d1[2], d1[0]*d2[2] - d2[0]*d1[2], d1[0]*d2[1] - d2[0]*d1[1]);

	if ((cross_prod[0]*cross_prod[0] + cross_prod[1]*cross_prod[1] + cross_prod[2]*cross_prod[2]) <= std::numeric_limits<double>::epsilon()) {
		return false;
	}

	return true;
}

bool GeometryUtilities::spansPlane(const std::vector<cv::Vec3d> &points)
{
	if (points.size() < 3) return false;
	
	// fix two points, use third point to determine colinearity
	for (unsigned int i = 2; i < points.size(); ++i) {
		if (spansPlane(points[0], points[1], points[i]))
			return true;
	}
	return false;
}

bool GeometryUtilities::isCounterClockwise(const cv::Vec2f p1, const cv::Vec2f p2, const cv::Vec2f p3)
{
	return clockwise(p1, p2, p3) > 0;
}

double GeometryUtilities::scalarProd(cv::Vec2f v1, cv::Vec2f v2)
{
	return v1[0]*v2[0]+v1[1]*v2[1];
}


/** clockwise gives an indicator for the relative position of 3 points:
- the points are colinear if clockwise == 0
- the points are in clockwise order (= describe a turn right) if clockwise > 0
- the points are in counterclockwise order if clockwise < 0
this is an extremely cheap operation that can be used as a sorting criterion for the graham scan algorithm
*/
int GeometryUtilities::clockwise(const cv::Vec2f idx1, const cv::Vec2f idx2, const cv::Vec2f idx3)
{
	double tmp = (idx2[0] - idx1[0]) * (idx3[1] - idx1[1]) - (idx2[1] - idx1[1]) * (idx3[0] - idx1[0]);
	if (fabs(tmp) < std::numeric_limits<float>::epsilon()) return 0;
	if (tmp > 0) return 1;
	return -1;
}

/** tests a point for being on an edge delimited by two points
 */
bool GeometryUtilities::pointOnEdge(const cv::Vec2f point, const cv::Vec2f edge_point1, const cv::Vec2f edge_point2) {
	// triangle? -> point not on the edge
	if (clockwise(point, edge_point1, edge_point2) != 0) return false;

	cv::Vec2f edge = edge_point1 - edge_point2;
	cv::Vec2f connection = edge_point1 - point;

	// oops, identical to first point?
	if ( (fabs(connection[0]) <= std::numeric_limits<float>::epsilon()) && (fabs(connection[1]) <= std::numeric_limits<float>::epsilon()) ) return true;

	// ratio < 1? -> point on edge
	if (fabs(connection[0]) > std::numeric_limits<float>::epsilon()) {
		double ratio = connection[0] / edge[0];
		return ((ratio >= 0) && (ratio <= 1));
	} else {
		// connection[1] must be non-zero now 
		double ratio = connection[1] / edge[1];
		return ((ratio >= 0) && (ratio <= 1));
	}
}

/** tests a point for being inside (or on the boundary of) a convex (!) polygon
 *  \param the test point
 *  \param the polygon, represented as a ordered sequence of edge points
 *  \return true if the point is in the polygon or on its boundary
 */
bool GeometryUtilities::pointInPolygon(const cv::Vec2f point, const std::vector<cv::Vec2f> &polygon) {
	unsigned int lastPolyPoint = polygon.size()-1;
	int indicator = clockwise(point, polygon[lastPolyPoint], polygon[0]);

	// point is on the line, but outside the edge of this polygon?
	if (indicator == 0) {
		return (pointOnEdge(point, polygon[lastPolyPoint], polygon[0])) ? true : false;
	}

	for (unsigned int i = 0; i < lastPolyPoint; ++i) {
		int tmp = clockwise(point, polygon[i], polygon[i+1]);
		if (tmp == 0) {
			// point is on the line, but outside the edge of this polygon?
			return (pointOnEdge(point, polygon[i], polygon[i+1])) ?  true : false;
		} else {
			// outside the poly if the point is clockwise with two points,
			// counterclockwise with others
			if (indicator != tmp) return false;
		}
	}
	return true;
}

}


