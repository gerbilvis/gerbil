/**
 *  \file   statistics.h
 *  \brief  A class for calculating basic statistics.
 *  \author Michael Bleier
 *  \date   2010
 */

#ifndef ILLUMESTIMATORS_COMMON_STATISTICS_H
#define ILLUMESTIMATORS_COMMON_STATISTICS_H

#include <opencv2/core/core.hpp>

namespace illumestimators {

/** The Statistics class provides functions for calculating basic statistics
 *  of images and vectors.
 */
class Statistics
{
public:
	/** Calculate the minimum of an image matrix.
	 *  \param image Input image matrix.
	 *  \return Vector of the minima of each color channel.
	 */
	static cv::Vec3d min(const cv::Mat_<cv::Vec3d>& image);
	/** Calculate the minimum of an image vector.
	 *  \param vector Input image vector.
	 *  \return Vector of the minima of each color channel.
	 */
	static cv::Vec3d min(const std::vector<cv::Vec3d>& vector);
	/** Calculate the minimum of a vector.
	 *  \param vector Input vector.
	 *  \return Minimum value of the vector.
	 */
	static double min(const std::vector<double>& vector);
	/** Calculate the minimum of a vector.
	 *  \param vector Input vector.
	 *  \return Minimum value of the vector.
	 */
	static double min(const cv::Vec3d& vector);

public:
	/** Calculate the maximum of an image matrix.
	 *  \param image Input image matrix.
	 *  \return Vector of the maxima of each color channel.
	 */
	static cv::Vec3d max(const cv::Mat_<cv::Vec3d>& image);
	/** Calculate the maximum of an image vector.
	 *  \param vector Input image vector.
	 *  \return Vector of the maxima of each color channel.
	 */
	static cv::Vec3d max(const std::vector<cv::Vec3d>& vector);
	/** Calculate the maximum of a vector.
	 *  \param vector Input vector.
	 *  \return Maximum value of the vector.
	 */
	static double max(const std::vector<double>& vector);
	/** Calculate the maximum of a vector.
	 *  \param vector Input vector.
	 *  \return Maximum value of the vector.
	 */
	static double max(const cv::Vec3d& vector);

public:
	/** Calculate the mean of an image matrix.
	 *  \param image Input image matrix.
	 *  \return Vector of the mean values of each color channel.
	 */
	static cv::Vec3d mean(const cv::Mat_<cv::Vec3d>& image);
	/** Calculate the mean of an image vector.
	 *  \param vector Input image vector.
	 *  \return Vector of the mean values of each color channel.
	 */
	static cv::Vec3d mean(const std::vector<cv::Vec3d>& vector);
	/** Calculate the mean of a vector.
	 *  \param vector Input vector.
	 *  \return Mean value of the vector.
	 */
	static double mean(const std::vector<double>& vector);
	/** Calculate the mean of a vector.
	 *  \param vector Input vector.
	 *  \return Mean value of the vector.
	 */
	static double mean(const cv::Vec3d& vector);

public:
	/** Calculate the standard deviation of an image matrix.
	 *  \param image Input image matrix.
	 *  \return Vector of the standard deviations of each color channel.
	 */
	static cv::Vec3d std(const cv::Mat_<cv::Vec3d>& image);
	/** Calculate the standard deviation of an image vector.
	 *  \param vector Input image vector.
	 *  \return Vector of the standard deviations of each color channel.
	 */
	static cv::Vec3d std(const std::vector<cv::Vec3d>& vector);
	/** Calculate the standard deviation of a vector.
	 *  \param vector Input vector.
	 *  \return Standard deviation of the vector.
	 */
	static double std(const std::vector<double>& vector);

public:
	/** Calculate the median of an image matrix.
	 *  \param image Input image matrix.
	 *  \return Vector of the median values of each color channel.
	 */
	static cv::Vec3d median(const cv::Mat_<cv::Vec3d>& image);
	/** Calculate the median of an image vector.
	 *  \param vector Input image vector.
	 *  \return Vector of the median values of each color channel.
	 */
	static cv::Vec3d median(const std::vector<cv::Vec3d>& vector);
	/** Calculate the median of a vector.
	 *  \param vector Input vector.
	 *  \return Median value of the vector.
	 */
	static double median(const std::vector<double>& vector);

public:
	/** Calculate the RMS of a vector.
	 *  \param vector Input vector.
	 *  \return RMS value of the vector.
	 */
	static double rms(const std::vector<double>& vector);
};

} // namespace illumestimators

#endif // ILLUMESTIMATORS_COMMON_STATISTICS_H
