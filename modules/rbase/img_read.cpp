#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "img_read.h"
#include "color_space.h"

namespace iread {

cv::Mat_<cv::Vec3d> IRead::img_read(const std::string &file_name, const std::string &colorspace, const int maxInt, const int verbosity)
{
	int tmpmax = maxInt;
	return img_read_dl_max(file_name, colorspace, 0, tmpmax, verbosity);
}

cv::Mat_<cv::Vec3d> IRead::img_read_dl(
	const std::string &file_name,
	const std::string &colorspace,
	const int darklevel,
	const int maxInt,
	const int verbosity)
{
	int tmpmax = maxInt;
	return img_read_dl_max(file_name, colorspace, darklevel, tmpmax, verbosity);
}

// return also maximum intensity
cv::Mat_<cv::Vec3d> IRead::img_read_dl_max(
	const std::string &file_name,
	const std::string &colorspace,
	const int darklevel,
	int &maxInt,
	const int verbosity)
{

	cv::Mat imageRaw = cv::imread(file_name, -1);
	cv::Mat_<cv::Vec3d> inputImage;

	if (imageRaw.type() == CV_16UC(3)) {
		cv::Mat_<cv::Vec3w> imageRawTyped = imageRaw;
		inputImage = cv::Mat_<cv::Vec3d>(imageRaw.rows, imageRaw.cols);

		double divisor;

		divisor = maxInt;
		if (divisor <= std::numeric_limits<double>::epsilon()) {
			divisor = getMax((cv::Mat_<cv::Vec3w>&)imageRawTyped, verbosity);
			maxInt = (int)divisor;
		}
		if (verbosity > 1) {
			std::cout << "max = " << maxInt << std::endl;
			std::cout << "darklevel = " << darklevel << ", divisor = " << divisor << std::endl;
		}

		for (int y = 0; y < imageRawTyped.rows; ++y) {
			for (int x = 0; x < imageRawTyped.cols; ++x) {
				cv::Vec3w &tmp = imageRawTyped[y][x];
				for (int c = 0; c < 3; ++c) {
					if (tmp[c] < darklevel) {
						inputImage[y][x][c] = 0;
					} else {
						inputImage[y][x][c] = (((double)tmp[c]) - darklevel) / divisor;
						if (inputImage[y][x][c] > 1) {
							inputImage[y][x][c] = 1;
						}
					}
				}
			}
		}

		if (verbosity > 0) std::cout << "16 bit" << std::endl;
		if (verbosity > 0) std::cout << "color space: " << colorspace << std::endl;
		if (colorspace.compare("rgb2srgb") == 0) {
			cv::Mat_<cv::Vec3d> result = ColorSpace::bgr2sbgr(inputImage);
			inputImage = result;
			divisor = getMax((cv::Mat_<cv::Vec3d>&) inputImage, verbosity);
			for (int y = 0; y < inputImage.rows; ++y) {
				for (int x = 0; x < inputImage.cols; ++x) {
					cv::Vec3d &tmp = inputImage[y][x];
					for (int c = 0; c < 3; ++c) {
						inputImage[y][x][c] = tmp[c] / divisor;
					}
				}
			}
		} else {
			if (colorspace.compare("srgb") == 0) {
				inputImage = ColorSpace::sbgr2bgr((cv::Mat_<cv::Vec3d>)inputImage);
			} // rgb: do nothing
		}

	} else { // 8 bit image

		if (verbosity > 0) std::cout << "8 bit" << std::endl;
		if (verbosity > 0) std::cout << "color space: " << colorspace << std::endl;

		if (colorspace.compare("srgb") == 0) {
			inputImage = ColorSpace::sbgr2bgr((cv::Mat_<cv::Vec3b>)imageRaw);
		} else {
			inputImage = ColorSpace::bgr2bgr((cv::Mat_<cv::Vec3b>)imageRaw);
		}
	}

	return inputImage;
}

int IRead::getMax(cv::Mat_<cv::Vec3b> &mat, int verbosity) {
	int max = -1;
	cv::Vec3d max3(-1, -1, -1);

	for (int y = 0; y < mat.rows; ++y)
		for (int x = 0; x < mat.cols; ++x)
			for (int c = 0; c < 3; ++c) {
				if (mat[y][x][c] > max)
					max = mat[y][x][c];
				if (mat[y][x][c] > max3[c]) max3[c] = mat[y][x][c];
			}

	if (verbosity > 0)
		std::cout << "max[0] = " << max3[0] << ", max[1] = " << max3[1] << ", max[2] = " << max3[2] << std::endl;

	return max;
}

int IRead::getMax(cv::Mat_<cv::Vec3w> &mat, int verbosity) {
	int max = -1;
	cv::Vec3d max3(-1, -1, -1);

	for (int y = 0; y < mat.rows; ++y)
		for (int x = 0; x < mat.cols; ++x)
			for (int c = 0; c < 3; ++c) {
				if (mat[y][x][c] > max)
					max = mat[y][x][c];
				if (mat[y][x][c] > max3[c]) max3[c] = mat[y][x][c];
			}

	if (verbosity > 0)
		std::cout << "max[0] = " << max3[0] << ", max[1] = " << max3[1] << ", max[2] = " << max3[2] << std::endl;
	return max;
}

double IRead::getMax(cv::Mat_<cv::Vec3d> &mat, int verbosity) {
	double max = -1;
	cv::Vec3d max3(-1, -1, -1);

	for (int y = 0; y < mat.rows; ++y)
		for (int x = 0; x < mat.cols; ++x)
			for (int c = 0; c < 3; ++c) {
				if (mat[y][x][c] > max)
					max = mat[y][x][c];
				if (mat[y][x][c] > max3[c]) max3[c] = mat[y][x][c];
			}

	if (verbosity > 0)
		std::cout << "max[0] = " << max3[0] << ", max[1] = " << max3[1] << ", max[2] = " << max3[2] << std::endl;
	return max;
}

}
