/*
	Copyright(c) 2012 Ralph Muessig	and Johannes Jordan
	<johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "som_tester.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

SOMTester::SOMTester(SOM &map, const multi_img &img)
	: som(map), image(img) { }

void SOMTester::getEdge(cv::Mat1d &dx, cv::Mat1d &dy)
{
	som.getEdge(image, dx, dy);
}

//SOMTester::SOMTester(const SOM &map, const multi_img &img, const vole::EdgeDetectionConfig &conf)
//	: som(map), image(img), config(conf),
//	  lookup(std::vector<std::vector<cv::Point> >(image.height,
//			 std::vector<cv::Point>(image.width)))
//{
//	for (int y = 0; y < image.height; y++) {
//		for (int x = 0; x < image.width; x++) {
//			lookup[y][x] = som.identifyWinnerNeuron(image(y, x));
//		}
//	}
//}

//void SOMTester::getEdge(cv::Mat1d &dx, cv::Mat1d &dy)
//{
//	std::cout << "Calculating derivatives (dx, dy)" << std::endl;

//	if (config.hack3d) {
//		getEdge3(dx, dy);
//		return;
//	}

//	dx = cv::Mat::zeros(image.height, image.width, CV_64F);
//	dy = cv::Mat::zeros(image.height, image.width, CV_64F);

//	// TODO: warum wird hier alles kopiert?
//	// collect SOM representant coordinates of all pixels
//	cv::Mat2d indices(image.height, image.width);
//	for (int y = 0; y < image.height; y++) {
//		cv::Vec2d *drow = indices[y];
//		for (int x = 0; x < image.width; x++) {
//			const cv::Point &p = lookup[y][x];
//			drow[x][0] = p.x;
//			drow[x][1] = p.y;
//		}
//	}

//	double maxIntensity = 0.0;

//	for (int y = 1; y < image.height-1; y++) {
//		double* x_ptr = dx[y];
//		double* y_ptr = dy[y];
//		cv::Vec2d *i_ptr = indices[y];
//		cv::Vec2d *i_uptr = indices[y-1];
//		cv::Vec2d *i_dptr = indices[y+1];
//		double xx, yy, valx, valy;

//		for (int x = 1; x < image.width-1; x++) {
//			{	// y-direction
//				xx = (i_uptr[x-1][0] + 2*i_uptr[x][0] + i_uptr[x+1][0]) * 0.25;
//				yy = (i_uptr[x-1][1] + 2*i_uptr[x][1] + i_uptr[x+1][1]) * 0.25;
//				cv::Point2d u(xx, yy);
//				xx = (i_dptr[x-1][0] + 2*i_dptr[x][0] + i_dptr[x+1][0]) * 0.25;
//				yy = (i_dptr[x-1][1] + 2*i_dptr[x][1] + i_dptr[x+1][1]) * 0.25;
//				cv::Point2d d(xx, yy);

//				valy = som.getDistance(u, d);
//				if (maxIntensity < valy)
//					maxIntensity = valy;

//				if ((u.x*u.x + u.y*u.y) < (d.x*d.x + d.y*d.y))
//					valy = -valy;
//				y_ptr[x] =  valy;
//			}
//			{	// x-direction
//				xx = (i_uptr[x-1][0] + 2*i_ptr[x-1][0] + i_dptr[x-1][0]) * 0.25;
//				yy = (i_uptr[x-1][1] + 2*i_ptr[x-1][1] + i_dptr[x-1][1]) * 0.25;
//				cv::Point2d u(xx, yy);
//				xx = (i_uptr[x+1][0] + 2*i_ptr[x+1][0] + i_dptr[x+1][0]) * 0.25;
//				yy = (i_uptr[x+1][1] + 2*i_ptr[x+1][1] + i_dptr[x+1][1]) * 0.25;
//				cv::Point2d d(xx, yy);

//				valx = som.getDistance(u, d);
//				if (maxIntensity < valx)
//					maxIntensity = valx;

//				if ((u.x*u.x + u.y*u.y) < (d.x*d.x + d.y*d.y))
//					valx = -valx;
//				x_ptr[x] = valx;
//			}
//		}
//	}

//	// normalization
//	cv::MatIterator_<double> ix, iy;
//	for (ix = dx.begin(), iy = dy.begin(); ix != dx.end(); ++ix, ++iy) {
//		// [-X .. X] -> [0 .. 1] but not really!! TODO
//		*ix = ((*ix + maxIntensity) * 0.5 / maxIntensity);
//		*iy = ((*iy + maxIntensity) * 0.5 / maxIntensity);
//	}
//}

//void SOMTester::getEdge3(cv::Mat1d &dx, cv::Mat1d &dy)
//{
//	std::cout << "Calculating derivatives (dx, dy)" << std::endl;

//	dx = cv::Mat::zeros(image.height, image.width, CV_64F);
//	dy = cv::Mat::zeros(image.height, image.width, CV_64F);

//	// collect SOM representant coordinates of all pixels
//	cv::Mat3d indices(image.height, image.width);
//	for (int y = 0; y < image.height; y++) {
//		cv::Vec3d *drow = indices[y];
//		for (int x = 0; x < image.width; x++) {
//			const cv::Point &p = lookup[y][x];
//			drow[x][0] = p.x;
//			drow[x][1] = p.y / som.getWidth();
//			drow[x][2] = p.y % som.getWidth();
//		}
//	}

//	double maxIntensity = 0.0;

//	for (int y = 1; y < image.height-1; y++) {
//		double* x_ptr = dx[y];
//		double* y_ptr = dy[y];
//		cv::Vec3d *i_ptr = indices[y];
//		cv::Vec3d *i_uptr = indices[y-1];
//		cv::Vec3d *i_dptr = indices[y+1];
//		double xx, yy, zz, valx, valy;

//		for (int x = 1; x < image.width-1; x++) {
//			{	// y-direction
//				xx = (i_uptr[x-1][0] + 2*i_uptr[x][0] + i_uptr[x+1][0]) * 0.25;
//				yy = (i_uptr[x-1][1] + 2*i_uptr[x][1] + i_uptr[x+1][1]) * 0.25;
//				zz = (i_uptr[x-1][2] + 2*i_uptr[x][2] + i_uptr[x+1][2]) * 0.25;
//				cv::Point3d u(xx, yy, zz);
//				xx = (i_dptr[x-1][0] + 2*i_dptr[x][0] + i_dptr[x+1][0]) * 0.25;
//				yy = (i_dptr[x-1][1] + 2*i_dptr[x][1] + i_dptr[x+1][1]) * 0.25;
//				zz = (i_dptr[x-1][2] + 2*i_dptr[x][2] + i_dptr[x+1][2]) * 0.25;
//				cv::Point3d d(xx, yy, zz);

//				valy = som.getDistance3(u, d);
//				if (maxIntensity < valy)
//					maxIntensity = valy;

//				// TODO ???
//				if ((u.x*u.x + u.y*u.y) < (d.x*d.x + d.y*d.y))
//					valy = -valy;
//				y_ptr[x] =  valy;
//			}
//			{	// x-direction
//				xx = (i_uptr[x-1][0] + 2*i_ptr[x-1][0] + i_dptr[x-1][0]) * 0.25;
//				yy = (i_uptr[x-1][1] + 2*i_ptr[x-1][1] + i_dptr[x-1][1]) * 0.25;
//				zz = (i_uptr[x-1][2] + 2*i_ptr[x-1][2] + i_dptr[x-1][2]) * 0.25;
//				cv::Point3d u(xx, yy, zz);
//				xx = (i_uptr[x+1][0] + 2*i_ptr[x+1][0] + i_dptr[x+1][0]) * 0.25;
//				yy = (i_uptr[x+1][1] + 2*i_ptr[x+1][1] + i_dptr[x+1][1]) * 0.25;
//				zz = (i_uptr[x+1][2] + 2*i_ptr[x+1][2] + i_dptr[x+1][2]) * 0.25;
//				cv::Point3d d(xx, yy, zz);

//				valx = som.getDistance3(u, d);
//				if (maxIntensity < valx)
//					maxIntensity = valx;

//				// TODO ???
//				if ((u.x*u.x + u.y*u.y) < (d.x*d.x + d.y*d.y))
//					valx = -valx;
//				x_ptr[x] = valx;
//			}
//		}
//	}

//	// normalization
//	cv::MatIterator_<double> ix, iy;
//	for (ix = dx.begin(), iy = dy.begin(); ix != dx.end(); ++ix, ++iy) {
//		// [-X .. X] -> [0 .. 1] but not really!! TODO
//		*ix = ((*ix + maxIntensity) * 0.5 / maxIntensity);
//		*iy = ((*iy + maxIntensity) * 0.5 / maxIntensity);
//	}
//}
