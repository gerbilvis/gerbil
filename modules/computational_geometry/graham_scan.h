#ifndef GRAHAM_SCAN_H
#define GRAHAM_SCAN_H

// computes the 2d-convex hull using the graham scan algorithm

#include "geometry_utilities.h"
#include "cv.h"
#include <vector>
#include <iostream>
#include <ostream>
#include <algorithm>
#include <cmath>


namespace vole {

class CmpScanCCWLower {
	public:
		CmpScanCCWLower(cv::Vec2f minPoint) : minPoint(minPoint) { }
		cv::Vec2f minPoint; 
		bool operator()(const cv::Vec2f &p2, const cv::Vec2f &p3) {
			int tmp = GeometryUtilities::clockwise(minPoint, p2, p3);
			if (tmp == 0) {
				return ((p2[0] - minPoint[0])*(p2[0] - minPoint[0]) + (p2[1] - minPoint[1])*(p2[1]-minPoint[1])) >
					   ((p3[0] - minPoint[0])*(p3[0] - minPoint[0]) + (p3[1] - minPoint[1])*(p3[1]-minPoint[1]));
			} else {
				return tmp > 0;
			}
		}
};

class GrahamScan {
public:
	GrahamScan();
	GrahamScan(cv::Mat_<unsigned char> debugImg);
	~GrahamScan();

	void fillDebugImg(const std::vector<cv::Vec2f > &hull);


	void getHull(std::vector<cv::Vec2f> &points, std::vector<cv::Vec2f> &hull);

private:
	cv::Mat_<unsigned char> debugImg;
	double minx;
	double miny;
	cv::Vec2f minPoint; 

	static bool isClockwise(const cv::Vec2f &p2, const cv::Vec2f &p3);
//	static bool isCounterClockwiseLower(const cv::Vec2f &p2, const cv::Vec2f &p3);
	static bool isCounterClockwise(const cv::Vec2f &p2, const cv::Vec2f &p3);

};

}

#endif // GRAHAM_SCAN_H

