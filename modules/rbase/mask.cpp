/**
 *  \file   mask.cpp
 *  \brief  A class for handling image masks.
 *  \author Michael Bleier
 *  \date   2010
 */

#include "mask.h"
#include <math.h>
#include <algorithm>

namespace iread {

std::vector<cv::Vec3d> Mask::maskedPixels(const cv::Mat_<cv::Vec3d>& image, const cv::Mat_<unsigned char>& mask)
{
	std::vector<cv::Vec3d> pixels;

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			if (mask(y, x) > 0) {
				pixels.push_back(image(y, x));
			}
		}
	}

	return pixels;
}

std::vector<cv::Vec3d> Mask::unmaskedPixels(const cv::Mat_<cv::Vec3d>& image, const cv::Mat_<unsigned char>& mask)
{
	std::vector<cv::Vec3d> pixels;

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			if (mask(y, x) == 0) {
				pixels.push_back(image(y, x));
			}
		}
	}

	return pixels;
}

std::vector<cv::Vec3d> Mask::maskedPixels(const cv::Mat_<cv::Vec3d>& image, const std::vector<cv::Point>& superpixel, const cv::Mat_<unsigned char>& mask)
{
	std::vector<cv::Vec3d> pixels;

	for(std::vector<cv::Point>::const_iterator it = superpixel.begin(); it != superpixel.end(); it++) {
		cv::Point coord = *it;

		if (mask(coord.y, coord.x) > 0) {
			pixels.push_back(image(coord.y, coord.x));
		}
	}

	return pixels;
}

std::vector<cv::Vec3d> Mask::unmaskedPixels(const cv::Mat_<cv::Vec3d>& image, const std::vector<cv::Point>& superpixel, const cv::Mat_<unsigned char>& mask)
{
	std::vector<cv::Vec3d> pixels;

	for(std::vector<cv::Point>::const_iterator it = superpixel.begin(); it != superpixel.end(); it++) {
		cv::Point coord = *it;

		if (mask(coord.y, coord.x) == 0) {
			pixels.push_back(image(coord.y, coord.x));
		}
	}

	return pixels;
}

void Mask::maskSaturatedPixels(const cv::Mat_<cv::Vec3d>& image, cv::Mat_<unsigned char>& mask, double threshold)
{
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			if (std::max(image(y, x)[0], std::max(image(y, x)[1], image(y, x)[2])) > threshold) {
				mask(y, x) = 255;
			}
		}
	}
}

void Mask::maskMinPixels(const cv::Mat_<cv::Vec3d>& image, cv::Mat_<unsigned char>& mask, double threshold)
{
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			if (std::min(image(y, x)[0], std::min(image(y, x)[1], image(y, x)[2])) < threshold) {
				mask(y, x) = 255;
			}
		}
	}
}

void Mask::maskDarkPixels(const cv::Mat_<cv::Vec3d>& image, cv::Mat_<unsigned char>& mask, double threshold)
{
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			if ((image(y, x)[0] + image(y, x)[1] + image(y, x)[2]) < threshold) {
				mask(y, x) = 255;
			}
		}
	}
}

void Mask::maskBorderPixels(const cv::Mat_<cv::Vec3d>& image, cv::Mat_<unsigned char>& mask, int width)
{
	if (width < 1) {
		return;
	}

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			if ((y < width) || (y >= image.rows - width) || (x < width) || (x >= image.cols - width)) {
				mask(y, x) = 255;
			}
		}
	}
}

} // namespace illumestimators
