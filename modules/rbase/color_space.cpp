/**
 *  \file   color_space.cpp
 *  \brief  A class for colorspace conversions.
 *  \author Michael Bleier
 *  \date   2010
 */

#include "color_space.h"

namespace iread {

illumestimators::Illum ColorSpace::calculateChromaticities(illumestimators::Illum color)
{
	const double sum = color.r + color.g + color.b;

	if (sum == 0) {
		return illumestimators::Illum();
	}
	return illumestimators::Illum(color.r / sum, color.g / sum, color.b / sum);
}


cv::Vec3d ColorSpace::calculateChromaticities(cv::Vec3d color)
{
	cv::Vec3d chromaticities;

	const double sum = color[0] + color[1] + color[2];

	if (sum == 0) {
		return chromaticities;
	}

	for (int i = 0; i < 3; i++) {
		chromaticities[i] = color[i] / sum;
	}

	return chromaticities;
}

cv::Mat_<cv::Vec3d> ColorSpace::colorCorrectImage(const cv::Mat_<cv::Vec3d>& image, const cv::Vec3d& estimate)
{
	cv::Mat_<cv::Vec3d> result = cv::Mat_<cv::Vec3d>(image.rows, image.cols);

	double max = 0;

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			for (int i = 0; i < 3; i++) {
				const double value = image(y, x)[i] / estimate[i];

				if (value > max) {
					max = value;
				}

				result(y, x)[i] = value;
			}
		}
	}

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			for (int i = 0; i < 3; i++) {
				result(y, x)[i] /= max;
			}
		}
	}

	return result;
}

/* computes from SRGB gamma'd image a linear BGR representation */
cv::Mat_<cv::Vec3d> ColorSpace::srgb2bgr(const unsigned char* input, int rows, int cols)
{
	cv::Mat_<cv::Vec3d> output = cv::Mat_<cv::Vec3d>(rows, cols);

	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			output(y, x)[0] = (double) input[(2 * rows * cols) + y + x * rows] / 255;
			output(y, x)[1] = (double) input[(1 * rows * cols) + y + x * rows] / 255;
			output(y, x)[2] = (double) input[(0 * rows * cols) + y + x * rows] / 255;

			for (int i = 0; i < 3; i++) {
				if (output(y, x)[i] <= 0.04045) {
					output(y, x)[i] /= 12.92;
				} else {
					output(y, x)[i] = pow((output(y, x)[i] + 0.055) / 1.055, 2.4);
				}
			}
		}
	}

	return output;
}

cv::Mat_<cv::Vec3d> ColorSpace::rgb2bgr(const unsigned char* input, int rows, int cols)
{
	cv::Mat_<cv::Vec3d> output = cv::Mat_<cv::Vec3d>(rows, cols);

	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			output(y, x)[0] = (double) input[(2 * rows * cols) + y + x * rows] / 255;
			output(y, x)[1] = (double) input[(1 * rows * cols) + y + x * rows] / 255;
			output(y, x)[2] = (double) input[(0 * rows * cols) + y + x * rows] / 255;
		}
	}

	return output;
}

cv::Mat_<cv::Vec3d> ColorSpace::sbgr2bgr(const cv::Mat_<cv::Vec3b>& input)
{
	cv::Mat_<cv::Vec3d> output = cv::Mat_<cv::Vec3d>(input.rows, input.cols);

	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			for (int i = 0; i < 3; i++) {
				const double value = input(y, x)[i] / 255.0;

				if (value <= 0.04045) {
					output(y, x)[i] = value / 12.92;
				} else {
					output(y, x)[i] = pow((value + 0.055) / 1.055, 2.4);
				}
			}
		}
	}

	return output;
}

cv::Mat_<cv::Vec3d> ColorSpace::bgr2bgr(const cv::Mat_<cv::Vec3b>& input)
{
	cv::Mat_<cv::Vec3d> output = cv::Mat_<cv::Vec3d>(input.rows, input.cols);

	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			for (int i = 0; i < 3; i++) {
				output(y, x)[i] = input(y, x)[i] / 255.0;
			}
		}
	}

	return output;
}

cv::Vec3d ColorSpace::rgb2bgr(const cv::Vec3d& input)
{
	cv::Vec3d output;

	output[0] = input[2];
	output[1] = input[1];
	output[2] = input[0];

	return output;
}

cv::Vec3d ColorSpace::bgr2rgb(const cv::Vec3d& input)
{
	cv::Vec3d output;

	output[0] = input[2];
	output[1] = input[1];
	output[2] = input[0];

	return output;
}

cv::Mat_<cv::Vec3d> ColorSpace::sbgr2bgr(const cv::Mat_<cv::Vec3d>& input)
{
	cv::Mat_<cv::Vec3d> output(input.rows, input.cols);

	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			for (int i = 0; i < 3; i++) {
				const double value = input(y, x)[i];

				if (value <= 0.04045) {
					output(y, x)[i] = value / 12.92;
				} else {
					output(y, x)[i] = pow((value + 0.055) / 1.055, 2.4);
				}
			}
		}
	}

	return output;
}

cv::Mat_<cv::Vec3d> ColorSpace::bgr2sbgr(const cv::Mat_<cv::Vec3d>& input)
{
	cv::Mat_<cv::Vec3d> output(input.rows, input.cols);

	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			for (int i = 0; i < 3; i++) {
				const double value = input[y][x][i];

				if (value <= 0.0031308) {
					output[y][x][i] = 12.92 * value;
				} else {
					output[y][x][i] = pow(1.055 * value, 1 / 2.4) - 0.055;
				}
			}
		}
	}
	return output;
}

} // namespace iread
