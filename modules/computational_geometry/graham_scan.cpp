#include "graham_scan.h"
#include "data_traits.h"

namespace vole {

	GrahamScan::GrahamScan() {
	}

	GrahamScan::GrahamScan(cv::Mat_<unsigned char> debugImg)
		: debugImg(debugImg)
	{
	}

	GrahamScan::~GrahamScan()
	{
	}

	// sets a point with intensity 64 on every hull point in the debugImg
	// note that this method assumes that debugImg is not null
	void GrahamScan::fillDebugImg(const std::vector<cv::Vec2f> &hull) {
		for (std::vector<cv::Vec2f>::const_iterator it = hull.begin(); it != hull.end(); it++) {
			int x = static_cast<int>(0.5 + (*it)[0]);
			int y = static_cast<int>(0.5 + (*it)[1]);
			if (x >= 0 && x < static_cast<int>(debugImg.cols) && y >= 0 && y < static_cast<int>(debugImg.rows))
				debugImg[y][x] = 64; // FIXME ist cv::Point-Reihenfolge x, y oder y, x?
				//debugImg.at(
		}
	}

	void GrahamScan::getHull(std::vector<cv::Vec2f> &points, std::vector<cv::Vec2f> &hull)
	{
	// find an extreme point (lowest y component (+ lowest x component, if necessary for disambiguation))
	hull = points;

	int numberElements = (int)hull.size();
	int minpos = -1;
	minx = INT_MAX; 
	miny = INT_MAX;
	for (int count = 0; count < numberElements; ++count) {
		if (hull[count][1] < miny) {
			minpos = count;
			miny = hull[count][1];
			minx = hull[count][0];
		} else {
			if ((fabs(hull[count][1] - miny) < DataTraits<float>::epsilon())) {
				if (hull[count][0] < minx) {
					minpos = count;
					minx = hull[count][0];
				}
			}
		}
	}
	// initial element goes to position 0;
	hull[minpos] = hull.front();
	hull.front()[0] = static_cast<float>(minx);
	hull.front()[1] = static_cast<float>(miny);
	minPoint[0] = static_cast<float>(minx);
	minPoint[1] = static_cast<float>(miny);

	// delete duplicates of the starting point, because they disturb the following sorting
	int skippedItems = 0;
	for (int count = 1; count < numberElements; ++count) {
		if ((fabs(hull[count][0] - minx) < DataTraits<float>::epsilon()) && (fabs(hull[count][1] - miny) < DataTraits<float>::epsilon())) {
			skippedItems++;
			// swap last (possibly valid) entry and this
			cv::Vec2f tmp = hull[numberElements-1];
			hull[numberElements-1] = hull[count];
			hull[count] = tmp;
			count--;
			numberElements--;
		}
	}
	if (skippedItems > 0) {
		hull.erase(hull.begin() + numberElements, hull.end());
	}
	
	// if only one, two or three points are left, we are almost done
	if (hull.size() < 4) {
		// In the case of 3 points: check whether 2 and 3 are equal.
		if ((hull.size() == 3)
			&& (fabs(hull[1][0] - hull[2][0]) < DataTraits<float>::epsilon())
			&& (fabs(hull[1][1] - hull[2][1]) < DataTraits<float>::epsilon())
		) {
			hull.pop_back();
		}
		if (debugImg.rows != 0) fillDebugImg(hull);
		return;
	}

	// sort all points by their angle (their tangents, respectively)
	CmpScanCCWLower my_cmp(minPoint);
	std::sort(
		hull.begin() + 1,
		hull.end(),
//		isCounterClockwiseLower
		my_cmp
	);

	// drop every point that is either left of two others or colinear with two others
	cv::Vec2f firstPoint = minPoint;
	cv::Vec2f middlePoint = hull[1];
	cv::Vec2f newPoint = hull[2];
	cv::Vec2f infinityPoint(DataTraits<float>::maxValue(), DataTraits<float>::maxValue());
	hull.push_back(hull[0]); // trick: the first point and the "infinityPoint" close the convex hull
	hull.push_back(infinityPoint); 

	int i = 3;
	int nPoints = 1; // the only point that is fixed is the starting point
	do
	{
		int res = GeometryUtilities::clockwise(firstPoint, middlePoint, newPoint);
		if (res == 0)	// if clockwise == 0 the points are colinear or double, therefore drop the one in the middle
						// first one has to find out which of them is in the middle, since sort might mix them up!
		{
			int d12 = static_cast<int>(GeometryUtilities::euclid_dist_square(firstPoint, middlePoint));
			int d13 = static_cast<int>(GeometryUtilities::euclid_dist_square(firstPoint, newPoint));
			int d23 = static_cast<int>(GeometryUtilities::euclid_dist_square(middlePoint, newPoint));

			// a point is dropped if it is either a duplicate or if it is between the two other points
			// drop middlePoint?
			if ((d12 == 0) || (d23 == 0) || ((d13 >= d12) && (d13 >= d23)))
			{
				middlePoint = newPoint;
				newPoint = hull[i];
				++i;
				// special case: if all points are on one line, the last remaining one is part of the solution
				if ((newPoint == hull[0]) && (nPoints == 1) && (hull[0] != middlePoint)) {
					hull[nPoints] == middlePoint;
					nPoints++;
				}
			} else {
				// drop newPoint?
				if ((d13 == 0) || ((d12 >= d13) && (d12 >= d23)))
				{
					newPoint = hull[i];
					++i;
					// special case: if all points are on one line, the last remaining one is part of the solution
					if ((newPoint == hull[0]) && (nPoints == 1) && (hull[0] != middlePoint)) {
						hull[nPoints] = middlePoint; 
						nPoints++;
					}

				} else {
					// drop the firstPoint (ugh)
					// replace firstPoint in the list of hull points with either middlePoint or newPoint
					// note that this can not happen for the starting Point, therefore it's safe to access nPoints-2
					if (GeometryUtilities::isCounterClockwise(hull[nPoints-2], middlePoint, newPoint))
					{
						firstPoint = middlePoint;
						hull[nPoints-1] = middlePoint;
						middlePoint = newPoint;
						newPoint = hull[i];
						++i;
					} else {
						// middlePoint remains the middlePoint, newPoint replaces the former firstPoint
						firstPoint = newPoint;
						hull[nPoints-1] = newPoint;
						newPoint = hull[i];
						++i;
					}
				} 
			}
		}
		else // if the points are in counterclockwise order, the middle point is part of the convex hull
		{
			if (res < 0)
				// if clockwise < 0 the points are in "right turn" order, which means that the middle point
				// can not be part of the convex hull -> drop the point in the middle.
				// Additionally, we have to check whether newPoint also invalidates further members of the convex hull.
			{
				for (int j = nPoints-1; (j >= 1) && !GeometryUtilities::isCounterClockwise(hull[j-1], hull[j], newPoint); j--)
				{
					nPoints--; // invalidates hull points that are inner points with respect to newPoint;
				}
				firstPoint = hull[nPoints-1];
				middlePoint = newPoint; // note that this point is not yet added to the hull,
				// since it might also become invalidated by the next point.
				newPoint = hull[i];
				++i;
			}
			else // clockwise > 0 indicates that (so far) the middle point is a member of the convex hull (-> the points are in "left turn" order)
			{
				hull[nPoints] = middlePoint;
				firstPoint = middlePoint;
				middlePoint = newPoint;
				newPoint = hull[i];
				++nPoints;
				++i;
			}
		}

	} while (newPoint != infinityPoint);

	// now crop the points array to fit exactly the convex hull;
	if (hull.back() == hull[0]) {
		hull.erase(hull.begin() + nPoints - 1, hull.end());
	} else {
		hull.erase(hull.begin() + nPoints, hull.end());
	}
	if (debugImg.rows > 0) fillDebugImg(hull);
//	CDebugUtils::dumpPoints(hull, std::string("C:\\Temp\\convex_hull_sorted.txt"));
	return;
	}


	bool GrahamScan::isClockwise(const cv::Vec2f &p2, const cv::Vec2f &p3)
	{
		cv::Vec2f minPoint(0, 0);
		return GeometryUtilities::clockwise(minPoint, p2, p3) < 0;
	}

/* Buggy: static minPoint at 0,0 breaks in some cases
	bool GrahamScan::isCounterClockwiseLower(const cv::Vec2f &p2, const cv::Vec2f &p3)
	{
		cv::Vec2f minPoint(0, 0);
		int tmp = GeometryUtilities::clockwise(minPoint, p2, p3);
		if (tmp == 0) {
			return ((p2[0] - minPoint[0])*(p2[0] - minPoint[0]) + (p2[1] - minPoint[1])*(p2[1]-minPoint[1])) >
				   ((p3[0] - minPoint[0])*(p3[0] - minPoint[0]) + (p3[1] - minPoint[1])*(p3[1]-minPoint[1]));
		} else {
			return tmp > 0;
		}
	}
*/

	bool GrahamScan::isCounterClockwise(const cv::Vec2f &p2, const cv::Vec2f &p3)
	{
		cv::Vec2f minPoint(0, 0);
		return GeometryUtilities::clockwise(minPoint, p2, p3) > 0;
	}


}
