#include "normal_image.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

namespace vole { namespace NormalImage {

cv::Mat_<cv::Vec3d> normalize(cv::Mat_<cv::Vec3b> img, int lowerLimit, int &upperLimit)
{
	if (upperLimit < 0) upperLimit = max_entry<cv::Vec3b>(img);
	// TODO: verbosity test necessary!
	std::cout << "byte version upper: " << upperLimit << std::endl;
	cv::Mat_<cv::Vec3d> scaled_image(img.rows, img.cols);
//	*#Q#($% convert to does not work :((
//	img.convertTo(scaled_image, 1.0/(upperLimit-lowerLimit), -(1.0*lowerLimit)/(upperLimit-lowerLimit));
	for (int y = 0; y < img.rows; ++y) {
		for (int x = 0; x < img.cols; ++x) {
			for (int c = 0; c < 3; ++c) {
				scaled_image[y][x][c] = ((double)img[y][x][c]-lowerLimit) / (double)(upperLimit - lowerLimit);
			}
		}
	}

	return scaled_image;
}

cv::Mat_<cv::Vec3d> normalizeClipped(cv::Mat_<cv::Vec3s> img, int lowerLimit, int &upperLimit)
{
	if (upperLimit < 0) upperLimit = max_entry<cv::Vec3s>(img);

	// TODO: verbosity test necessary!
	std::cout << "short version upper: " << upperLimit << ", lower: " << lowerLimit << std::endl;

	cv::Mat_<cv::Vec3d> scaled_image(img.rows, img.cols);
//	*#Q#($% convert to does not work :((
//	img.convertTo(scaled_image, 1.0/(upperLimit-lowerLimit), -(1.0*lowerLimit)/(upperLimit-lowerLimit));
	for (int y = 0; y < img.rows; ++y) {
		for (int x = 0; x < img.cols; ++x) {
			for (int c = 0; c < 3; ++c) {
				scaled_image[y][x][c] = ((double)img[y][x][c]-lowerLimit) / (double)(upperLimit - lowerLimit);
				if (scaled_image[y][x][c] < 0)
					scaled_image[y][x][c] = 0;
				else if (scaled_image[y][x][c] == 1) {
					scaled_image[y][x][c] = 1;
				}
			}
		}
	}
	return scaled_image;
}

cv::Mat_<cv::Vec3d> normalize(cv::Mat_<cv::Vec3s> img, int lowerLimit, int &upperLimit)
{
	if (upperLimit < 0) upperLimit = max_entry<cv::Vec3s>(img);

	// TODO: verbosity test necessary!
	std::cout << "short version upper: " << upperLimit << ", lower: " << lowerLimit << std::endl;

	cv::Mat_<cv::Vec3d> scaled_image(img.rows, img.cols);
//	*#Q#($% convert to does not work :((
//	img.convertTo(scaled_image, 1.0/(upperLimit-lowerLimit), -(1.0*lowerLimit)/(upperLimit-lowerLimit));
	for (int y = 0; y < img.rows; ++y) {
		for (int x = 0; x < img.cols; ++x) {
			for (int c = 0; c < 3; ++c) {
				scaled_image[y][x][c] = ((double)img[y][x][c]-lowerLimit) / (double)(upperLimit - lowerLimit);
			}
		}
	}
	return scaled_image;
}

#ifdef WITH_QT
cv::Mat_<cv::Vec3d> normalize(QImage *img, int lowerLimit, int &upperLimit)
{
	if (upperLimit < 0) upperLimit = max_entry(img);

	int rows = img->height();
	int cols = img->width();
	cv::Mat_<cv::Vec3d> scaled_image(rows, cols);
	double divisor = upperLimit - lowerLimit;
	double align0 = 1.0*lowerLimit / divisor;
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			const QRgb point = img->pixel(x, y);
			scaled_image[y][x] = cv::Vec3d(
				qBlue(point)/divisor - align0,
				qGreen(point)/divisor - align0,
				qRed(point)/divisor - align0
			);
		}
	}
	cv::imwrite("/tmp/testme.png", scaled_image*255);
	return scaled_image;
}


// assume always three channels
double max_entry(QImage *img)
{
	int rows = img->height();
	int cols = img->width();
	double max = -1;
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			const QRgb point = img->pixel(x, y);
			if (qBlue(point) > max) max = qBlue(point);
			if (qGreen(point) > max) max = qGreen(point);
			if (qRed(point) > max) max = qRed(point);
		}
	}
	return max;
}

#endif // WITH_QT

} // NormalImage
} // vole
