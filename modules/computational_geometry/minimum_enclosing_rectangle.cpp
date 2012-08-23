#include "minimum_enclosing_rectangle.h"
#include "geometry_utilities.h"
#include "global_defines.h"
#include <limits>

namespace vole {

	MinimumEnclosingRectangle::MinimumEnclosingRectangle()
	{
	}

	MinimumEnclosingRectangle::MinimumEnclosingRectangle(cv::Mat_<unsigned char> debugImg)
		: debugImg(debugImg)
	{
	}

	MinimumEnclosingRectangle::~MinimumEnclosingRectangle()
	{
	}

//	typedef ContainerTraits<T> traits_type;

	/*
	void MinimumEnclosingRectangle::convertToFloatVector(
		const std::vector<cv::Vec2f> &input, std::vector<cv::Vec2f> &output)
	{
		output.clear();
		output.resize(input.size());
		for (unsigned int i = 0; i < input.size(); i++) {
			output[i] = cv::Vec2f(static_cast<float>(input[i][0]), static_cast<float>(input[i][1]));
		}
	}

	void MinimumEnclosingRectangle::convertToFloatVector(const std::vector<cv::Vec2f > &input, std::vector<cv::Vec2f > &output) {
		output.clear();
		output.resize(input.size());
		for (unsigned int i = 0; i < input.size(); i++) {
			output[i] = cv::Vec2f(static_cast<float>(input[i][0]), static_cast<float>(input[i][1]));
		}
	}
*/

	std::vector<cv::Vec2f > &MinimumEnclosingRectangle::getPoints()
	{
		return points;
	}

	/** * assumes to obtain a ordered list of points that form the convex hull of
	* a polygon in counter-clockwise order.
	* everything else will break the code in getMinimumEnclosingRectangle. Use CConvexHull to convert an arbitrary
	* point cloud into one satisfying the demands of getMinimumEnclosingRectangle.
	*/
	void MinimumEnclosingRectangle::setPoints(std::vector<cv::Vec2f > &newPoints)
	{
		points = newPoints;
	}

	void MinimumEnclosingRectangle::fillDebugImg(const std::vector<cv::Vec2f> merPts) {
		for (std::vector<cv::Vec2f >::const_iterator it = merPts.begin(); it != merPts.end(); it++) {
			int x = static_cast<int>(0.5 + (*it)[0]);
			int y = static_cast<int>(0.5 + (*it)[1]);
			if ((x >= 0) && (x < debugImg.cols) && (y >= 0) && (y < debugImg.rows))
				debugImg[y][x] = 128 | debugImg[y][x];
		}
	}

	/** 
	 * \return index of the point and distance
	 */
	std::pair<int, double> MinimumEnclosingRectangle::find_most_distant_point_from_line(int index, cv::Vec2f direction) {
		cv::Vec2f intersection;
		int best_index = index;
		double best_dist_square = 0;
		for (unsigned int i = 1; i < points.size(); ++i) {
			int ii = (index+i) % points.size();
			GeometryUtilities::raisePerpendicular(	points[index], direction, points[ii], intersection );
			double tmp_dist_square = GeometryUtilities::distSquare(points[ii], intersection);
			if (tmp_dist_square < best_dist_square + DataTraits<double>::epsilon())
				return std::pair<int, double>(best_index, std::sqrt(best_dist_square));
			best_dist_square = tmp_dist_square;
			best_index = ii;
		}
		return std::pair<int, double>(best_index, std::sqrt(best_dist_square));
	}

	/** Finds the two points that are farthest away when projected on a line defined by index1 and index2
	 * \param index1 
	 * \return indices of the extremal points
	 */
	std::pair<int, int> MinimumEnclosingRectangle::find_extremal_points_from_line(int index1, int index2, double *dist) {
		double offset_to_index;
		cv::Vec2f direction = points[index2] - points[index1];
		cv::Vec2f intersection, best_i1, best_i2;
		std::pair<int, int> extremal_points(-1, -1);
		std::pair<double, double> dist_extremal_points(std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity());
		for (unsigned int i = 0; i < points.size(); ++i) {
			GeometryUtilities::raisePerpendicular(	points[index1], direction, points[i], intersection );
			if (direction[0] != 0) {
				offset_to_index = points[index1][0] - intersection[0];
			} else {
				offset_to_index = points[index1][1] - intersection[1];
			}
			if (offset_to_index < dist_extremal_points.first) {
				dist_extremal_points.first = offset_to_index;
				extremal_points.first = i;
				best_i1 = intersection;
			}
			if (offset_to_index > dist_extremal_points.second) {
				dist_extremal_points.second = offset_to_index;
				extremal_points.second = i;
				best_i2 = intersection;
			}
		}
		if (dist != NULL) *dist = std::sqrt(GeometryUtilities::distSquare(best_i1, best_i2));
		return extremal_points;
	}

	/**
	 */
	double MinimumEnclosingRectangle::check_one_rectangle(int index1, int index2, double largestArea, int *tmp_result) {
		cv::Vec2f direction = points[index2] - points[index1];
		double dist;
		// establish to parallel lines that enclose the rectangle
		std::pair<int, double> most_distant_point = find_most_distant_point_from_line(index2, direction);
		std::pair<int, int> extremal_points = find_extremal_points_from_line(index1, index2, &dist);

		// the two perpendicular distances of the outmost points, we can directly compute the area
		double tmpArea = dist * most_distant_point.second;
		if (tmpArea < largestArea) {
			tmp_result[0] = index2;
			tmp_result[1] = most_distant_point.first;
			tmp_result[2] = extremal_points.first;
			tmp_result[3] = extremal_points.second;
			return tmpArea;
		}
		return largestArea;
	}

	//	void MinimumEnclosingRectangle::registerResult(std::vector<cv::Vec2f> &result, cv::Vec2f p1, cv::Vec2f p2, cv::Vec2f p3, cv::Vec2f 

	/** Computes the minimum area enclosing rectangle of the points given in
	 * setPoints. Note that this implementation is O(n^2) (!), although in
	 * theory this would not be necessary (see and debug
	 * getMinimumEnclosingRectangle for reference)
	*/
	void MinimumEnclosingRectangle::getMerOSquared(std::vector<cv::Vec2f> &result) {
		result.clear();
		result.push_back(cv::Vec2f()); result.push_back(cv::Vec2f());
		result.push_back(cv::Vec2f()); result.push_back(cv::Vec2f());
		int tmp_result[4];
		tmp_result[0] = tmp_result[1] = tmp_result[2] = tmp_result[3] = 0;
		double largestArea = check_one_rectangle(points.size()-1, 0, DBL_MAX, tmp_result);
		// for every edge - find the most distant edge point, compute the area (O(n^2))
		for (unsigned int i = 1; i < points.size(); ++i) {
			largestArea = check_one_rectangle(i-1, i, largestArea, tmp_result);
		}
		// convert tmp_result in result points.
		// tmp_result contains the indices of the extremal points.
		cv::Vec2f direction = points[tmp_result[0]] - points[(points.size() + tmp_result[0]-1) % points.size()];
		cv::Vec2f direction_perp(direction[0], -direction[1]);
		GeometryUtilities::raisePerpendicular( points[tmp_result[0]], direction, points[tmp_result[3]], result[0]);
		GeometryUtilities::raisePerpendicular( points[tmp_result[0]], direction, points[tmp_result[2]], result[1]);
		GeometryUtilities::raisePerpendicular( points[tmp_result[1]], direction, points[tmp_result[2]], result[2]);
		GeometryUtilities::raisePerpendicular( points[tmp_result[1]], direction, points[tmp_result[3]], result[3]);
	}

	/** BUGGY VERSION! Computes the minimum area enclosing rectangle of the points given in setPoints in O(n)
	*/
	void MinimumEnclosingRectangle::getMinimumEnclosingRectangle(std::vector<cv::Vec2f> &result) {
		// Strategie:
		// Fuer ein konvexes Polygon gilt: Mindestens eine gueltige Loesung besitzt eine gemeinsame Kante mit dem Polygon.
		// Also genuegt es, fuer alle Kanten des Polygons zu pruefen, ob sie eine minimale Loesung bieten.
		// Das geht in O(n), weil in der "idealen" Implementierung jede Kante einmal betrachtet wird.
		// Die vorliegende Loesung ist um einen konstanten Faktor langsamer, weil Symmetrien nicht beruecksichtigt werden,
		// aber das sollte keine signifikante Rolle spielen.

		result.clear();
		int c_pos[4], i;
		c_pos[0] = c_pos[1] = c_pos[2] = c_pos[3] = -1;
		int curAnglePos = 0;
		cv::Vec2f directions[] = {cv::Vec2f(-1, 0), cv::Vec2f(0, 1), cv::Vec2f(1, 0), cv::Vec2f(0, -1)};
		float angles[4], minx, miny, maxx, maxy;
		double minArea, usedAngle, curAngle;
		int nPoints = static_cast<int>(points.size());

		// find minimum and maximum x- and y-coordinates in the given polygon
		// this is also a first approximation for the minimum enclosing rectangle
		minx = miny = DataTraits<float>::maxValue();
		maxx = maxy = DataTraits<float>::minValue();
		for (i = 0; i < nPoints; i++)
		{
			if (points[i][0] < minx) { minx = points[i][0]; c_pos[0] = i; }
			if (points[i][1] < miny) { miny = points[i][1]; c_pos[1] = i; }
			if (points[i][0] > maxx) { maxx = points[i][0]; c_pos[2] = i; }
			if (points[i][1] > maxy) { maxy = points[i][1]; c_pos[3] = i; }
		}
		result.push_back(cv::Vec2f(minx, miny));
		result.push_back(cv::Vec2f(maxx, miny));
		result.push_back(cv::Vec2f(maxx, maxy));
		result.push_back(cv::Vec2f(minx, maxy));
		minArea = (maxx - minx)*(maxy - miny);

		// for every extremal point, compute the angle between the initialization (coordinate axis)
		// and the next point on the convex hull of the polygon. Note that this definitely breaks if the
		// "convex hull" contains non-convex segments.
		curAngle = DataTraits<float>::maxValue();
		for (i = 0; i < 4; i++) {
			cv::Vec2f v(points[(c_pos[i]+1)%nPoints] - points[c_pos[i]]);
			double tmpAngle = std::acos(
				GeometryUtilities::scalarProd(directions[i], v)) /
					(std::sqrt(
						GeometryUtilities::scalarProd(directions[i], directions[i]) *
						GeometryUtilities::scalarProd(v, v)
						)
					); 
			// most likely can the following line be deleted. Remember to delete also the same line below then.
			if (tmpAngle > VOLE_PI) tmpAngle -= VOLE_PI;
			tmpAngle = std::abs(tmpAngle);
			if (tmpAngle < curAngle) {
				curAngle = tmpAngle;
				curAnglePos = i;
			}
			angles[i] = static_cast<float>(tmpAngle);
		}

		usedAngle = 0;
		while (usedAngle < VOLE_PI_2) {
			cv::Vec2f cutPosNext, cutPosPrev, cutPosPrev2;
			// rotate the line at curAnglePos and calculate the new area

			// update the direction of the selected line
			directions[curAnglePos] = points[(c_pos[curAnglePos]+1) % nPoints] - points[c_pos[curAnglePos]];

			std::cout << "----" << std::endl;

			// compute the area of this rectangle: three points are needed
			GeometryUtilities::raisePerpendicular(	points[c_pos[curAnglePos]], directions[curAnglePos], points[c_pos[(curAnglePos+1)%4]], cutPosNext );
			std::cout << "senkrechte auf [" << points[c_pos[curAnglePos]][0] << "," << points[c_pos[curAnglePos]][1] <<
				"] in Richtung [" << directions[curAnglePos][0] << "," << directions[curAnglePos][1] <<
				"] von Punkt ["   << points[c_pos[(curAnglePos+1)%4]][0] << "," << points[c_pos[(curAnglePos+1)%4]][0] <<
				"] ----> ["       << cutPosNext[0] << "," << cutPosNext[1] << "]" << std::endl;

			GeometryUtilities::raisePerpendicular(	points[c_pos[curAnglePos]], directions[curAnglePos], points[c_pos[(curAnglePos+3)%4]], cutPosPrev );
			std::cout << "senkrechte auf [" << points[c_pos[curAnglePos]][0] << "," << points[c_pos[curAnglePos]][1] <<
				"] in Richtung [" << directions[curAnglePos][0] << "," << directions[curAnglePos][1] <<
				"] von Punkt ["   << points[c_pos[(curAnglePos+3)%4]][0] << "," << points[c_pos[(curAnglePos+3)%4]][0] <<
				"] ----> ["       << cutPosPrev[0] << "," << cutPosPrev[1] << "]" << std::endl;

			GeometryUtilities::raisePerpendicular(	points[c_pos[(curAnglePos+2)%4]], directions[curAnglePos], points[c_pos[(curAnglePos+3)%4]], cutPosPrev2 );
			std::cout << "senkrechte auf [" << points[c_pos[(curAnglePos+2)%4]][0] << "," << points[c_pos[(curAnglePos+2)%4]][1];
			std::cout << "] in Richtung [" << directions[curAnglePos][0] << "," << directions[curAnglePos][1];
			std::cout << "] von Punkt ["   << points[c_pos[(curAnglePos+3)%4]][0] << "," << points[c_pos[(curAnglePos+3)%4]][0];
			std::cout << "] ----> ["       << cutPosPrev2[0] << "," << cutPosPrev2[1] << "]" << std::endl;

			cv::Vec2f t1 = cutPosPrev2 - cutPosPrev;
			cv::Vec2f t2 = cutPosNext  - cutPosPrev;
			double tmpArea = std::abs(parallelpipedArea(t1, t2));

			std::cout << "tmpArea: " << minArea << std::endl;

			if (tmpArea < minArea) {
				std::cout << "-->MIN!" << std::endl;
				minArea = tmpArea;
				result[0] = cutPosPrev2;
				result[1] = cutPosPrev;
				result[2] = cutPosNext;
				// compute fourth point
				GeometryUtilities::raisePerpendicular(
					points[c_pos[(curAnglePos+2)%4]], directions[curAnglePos], points[c_pos[(curAnglePos+1)%4]], 
					result[3]);
			}

			// update the angle at curAnglePos and subtract this angle from all the other angles;
			for (i = 0; i < 4; i++) {
				if (i == curAnglePos) {
					c_pos[i] = (c_pos[i]+1) % nPoints;
					cv::Vec2f v = cv::Vec2f(points[(c_pos[i]+1)%nPoints] - points[c_pos[i]]);
					double tmpAngle = std::acos(
						GeometryUtilities::scalarProd(directions[i], v) /
						std::sqrt(
							GeometryUtilities::scalarProd(directions[i], directions[i]) *
							GeometryUtilities::scalarProd(v, v)
						)
					);
					if (tmpAngle > VOLE_PI) tmpAngle -= VOLE_PI;
					usedAngle += angles[i];
					angles[i] = static_cast<float>(tmpAngle);
					// update the direction of the selected line
					directions[curAnglePos] = v;
				} else {
					angles[i] -= static_cast<float>(curAngle);
				}
			}
			// now look for the new minimum angle
			curAngle = DataTraits<float>::maxValue();
			for (i = 0; i < 4; i++) {
				if (angles[i] < curAngle) {
					curAngle = angles[i];
					curAnglePos = i;
				}
			}
		}
		if (debugImg.rows > 0) fillDebugImg(result);
	}

	double MinimumEnclosingRectangle::parallelpipedArea(cv::Vec2f &v1, cv::Vec2f &v2)
	{
		double x3;
		// erst Kreuzprodukt der beiden Punkte berechnen, dann Skalarprodukt; in 2D sollte es das sein.
		x3 =  v1[0]*v2[1] - v1[1]*v2[0];
		return x3;
	}

}

