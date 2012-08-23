/**
 *  \file   statistics.cpp
 *  \brief  A class for calculating statistics.
 *  \author Michael Bleier
 *  \date   2010
 */

#include "vole_statistics.h"
#include <limits>

namespace illumestimators {

cv::Vec3d Statistics::min(const cv::Mat_<cv::Vec3d>& image)
{
	cv::Vec3d min = cv::Vec3d(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());

	for (cv::Mat_<cv::Vec3d>::const_iterator it = image.begin(); it != image.end(); it++) {
		for (int i = 0; i < 3; i++) {
			if ((*it)[i] < min[i]) {
				min[i] = (*it)[i];
			}
		}
	}

	return min;
}

cv::Vec3d Statistics::min(const std::vector<cv::Vec3d>& vector)
{
	cv::Vec3d min = cv::Vec3d(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());

	for (std::vector<cv::Vec3d>::const_iterator it = vector.begin(); it != vector.end(); it++) {
		for (int i = 0; i < 3; i++) {
			if ((*it)[i] < min[i]) {
				min[i] = (*it)[i];
			}
		}
	}

	return min;
}

double Statistics::min(const std::vector<double>& vector)
{
	double min = std::numeric_limits<double>::max();

	for (std::vector<double>::const_iterator it = vector.begin(); it != vector.end(); it++) {
		if (*it < min) {
			min = *it;
		}
	}

	return min;
}

double Statistics::min(const cv::Vec3d& vector)
{
	double min = std::numeric_limits<double>::max();

	for (int i = 0; i < 3; i++) {
		if (vector[i] < min) {
			min = vector[i];
		}
	}

	return min;
}

cv::Vec3d Statistics::max(const cv::Mat_<cv::Vec3d>& image)
{
	cv::Vec3d max = cv::Vec3d(-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max());

	for (cv::Mat_<cv::Vec3d>::const_iterator it = image.begin(); it != image.end(); it++) {
		for (int i = 0; i < 3; i++) {
			if ((*it)[i] > max[i]) {
				max[i] = (*it)[i];
			}
		}
	}

	return max;
}

cv::Vec3d Statistics::max(const std::vector<cv::Vec3d>& vector)
{
	cv::Vec3d max = cv::Vec3d(-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max());

	for (std::vector<cv::Vec3d>::const_iterator it = vector.begin(); it != vector.end(); it++) {
		for (int i = 0; i < 3; i++) {
			if ((*it)[i] > max[i]) {
				max[i] = (*it)[i];
			}
		}
	}

	return max;
}

double Statistics::max(const std::vector<double>& vector)
{
	double max = -std::numeric_limits<double>::max();

	for (std::vector<double>::const_iterator it = vector.begin(); it != vector.end(); it++) {
		if (*it > max) {
			max = *it;
		}
	}

	return max;
}

double Statistics::max(const cv::Vec3d& vector)
{
	double max = -std::numeric_limits<double>::max();

	for (int i = 0; i < 3; i++) {
		if (vector[i] > max) {
			max = vector[i];
		}
	}

	return max;
}

cv::Vec3d Statistics::mean(const cv::Mat_<cv::Vec3d>& image)
{
	cv::Vec3d mean = cv::Vec3d(0, 0, 0);

	for (cv::Mat_<cv::Vec3d>::const_iterator it = image.begin(); it != image.end(); it++) {
		for (int i = 0; i < 3; i++) {
			mean[i] += (*it)[i];
		}
	}

	const int size = image.rows * image.cols;

	for (int i = 0; i < 3; i++) {
		mean[i] /= size;
	}

	return mean;
}

cv::Vec3d Statistics::mean(const std::vector<cv::Vec3d>& vector)
{
	cv::Vec3d mean = cv::Vec3d(0, 0, 0);

	for (std::vector<cv::Vec3d>::const_iterator it = vector.begin(); it != vector.end(); it++) {
		for (int i = 0; i < 3; i++) {
			mean[i] += (*it)[i];
		}
	}

	const int size = vector.size();

	for (int i = 0; i < 3; i++) {
		mean[i] /= size;
	}

	return mean;
}

double Statistics::mean(const std::vector<double>& vector)
{
	double mean = 0;

	for (std::vector<double>::const_iterator it = vector.begin(); it != vector.end(); it++) {
		mean += *it;
	}

	mean /= vector.size();

	return mean;
}

double Statistics::mean(const cv::Vec3d& vector)
{
	double mean = 0;

	for (int i = 0; i < 3; i++) {
		mean += vector[i];
	}

	mean /= 3;

	return mean;
}

cv::Vec3d Statistics::std(const cv::Mat_<cv::Vec3d>& image)
{
	cv::Vec3d std = cv::Vec3d(0, 0, 0);
	cv::Vec3d m = mean(image);

	for (cv::Mat_<cv::Vec3d>::const_iterator it = image.begin(); it != image.end(); it++) {
		for (int i = 0; i < 3; i++) {
			std[i] += pow((*it)[i] - m[i], 2);
		}
	}

	const int size = image.rows * image.cols;

	for (int i = 0; i < 3; i++) {
		std[i] = sqrt(std[i] / size);
	}

	return std;
}

cv::Vec3d Statistics::std(const std::vector<cv::Vec3d>& vector)
{
	cv::Vec3d std = cv::Vec3d(0, 0, 0);
	cv::Vec3d m = mean(vector);

	for (std::vector<cv::Vec3d>::const_iterator it = vector.begin(); it != vector.end(); it++) {
		for (int i = 0; i < 3; i++) {
			std[i] += pow((*it)[i] - m[i], 2);
		}
	}

	const int size = vector.size();

	for (int i = 0; i < 3; i++) {
		std[i] = sqrt(std[i] / size);
	}

	return std;
}

double Statistics::std(const std::vector<double>& vector)
{
	double std = 0;
	double m = mean(vector);

	for (std::vector<double>::const_iterator it = vector.begin(); it != vector.end(); it++) {
		std += pow((*it) - m, 2);
	}

	std = sqrt(std / vector.size());

	return std;
}

cv::Vec3d Statistics::median(const cv::Mat_<cv::Vec3d>& image)
{
	std::vector<double> v[3];

	for (cv::Mat_<cv::Vec3d>::const_iterator it = image.begin(); it != image.end(); it++) {
		for (int i = 0; i < 3; i++) {
			v[i].push_back((*it)[i]);
		}
	}

	return cv::Vec3d(median(v[0]), median(v[1]), median(v[2]));
}

cv::Vec3d Statistics::median(const std::vector<cv::Vec3d>& vector)
{
	std::vector<double> v[3];

	for (std::vector<cv::Vec3d>::const_iterator it = vector.begin(); it != vector.end(); it++) {
		for (int i = 0; i < 3; i++) {
			v[i].push_back((*it)[i]);
		}
	}

	return cv::Vec3d(median(v[0]), median(v[1]), median(v[2]));
}

double Statistics::median(const std::vector<double>& vector)
{
	if (vector.size() < 1) {
		return 0;
	}

	std::vector<double> v = vector;
	sort(v.begin(), v.end());

	return v[vector.size() / 2];
}

double Statistics::rms(const std::vector<double>& vector)
{
	double meansquared = 0;

	for (std::vector<double>::const_iterator it = vector.begin(); it != vector.end(); it++) {
		meansquared += pow(*it, 2);
	}

	meansquared /= vector.size();

	return sqrt(meansquared);
}

} // namespace illuminantestimators
