//! \file CMinimumEnclosingRectangle.h Computes the minimum area enclosing rectangle around a convex hull.

#ifndef VOLE_MINIMUM_ENCLOSING_RECTANGLE_H_
#define VOLE_MINIMUM_ENCLOSING_RECTANGLE_H_

#include "data_traits.h"

#include "cv.h"

#include <vector>
#include <iostream>
#include <ostream>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <climits>


// #include "graham_scan.h"


namespace vole {

class MinimumEnclosingRectangle
{
public:
	MinimumEnclosingRectangle();
	MinimumEnclosingRectangle(cv::Mat_<unsigned char> debugImg);

	~MinimumEnclosingRectangle();

	/* REMOVE ME
	void convertToFloatVector(const std::vector<cv::Vec2f> &input, std::vector<cv::Vec2f> &output);
	void convertToFloatVector(const std::vector<cv::Vec2f> &input, std::vector<cv::Vec2f> &output);
	*/

	std::vector<cv::Vec2f > &getPoints();

	/*
	* setPoints assumes to obtain a ordered list of points that form the convex hull of
	* a polygon in counter-clockwise order.
	* everything else will break the code in getMinimumEnclosingRectangle. Use CConvexHull to convert an arbitrary
	* point cloud into one satisfying the demands of getMinimumEnclosingRectangle.
	*/
	void setPoints(std::vector<cv::Vec2f > &newPoints);

	void fillDebugImg(const std::vector<cv::Vec2f> merPts);

	void getMerOSquared(std::vector<cv::Vec2f> &result);

	/** FIXME buggy version!
	* Computes the minimum area enclosing rectangle of the points given in setPoints
	*/
	void getMinimumEnclosingRectangle(std::vector<cv::Vec2f> &result);

private:
	cv::Mat_<unsigned char> debugImg;
	std::vector<cv::Vec2f> points;
	double parallelpipedArea(cv::Vec2f &v1, cv::Vec2f &v2);

	double check_one_rectangle(int index1, int index2, double largestArea, int *tmp_result);
	std::pair<int, int> find_extremal_points_from_line(int index1, int index2, double *dist);
	std::pair<int, double> find_most_distant_point_from_line(int index, cv::Vec2f direction);
};

}

#endif // VOLE_MINIMUM_ENCLOSING_RECTANGLE_H_

