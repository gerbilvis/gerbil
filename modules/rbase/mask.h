/**
 *  \file   mask.h
 *  \brief  A class for handling image masks.
 *  \author Michael Bleier
 *  \date   2010
 */

#ifndef IREAD_COMMON_MASK_H
#define IREAD_COMMON_MASK_H

#include <opencv2/core/core.hpp>
#include <vector>

namespace iread {

/** The mask class provides helper functions for handling image masks.
 *  Masks are stored as matrices of the type unsigned char.
 *  Values greater than zero indicate masked pixels.
 */
class Mask
{
public:
	/** Extract masked pixels from an image.
	 *  \param image Input image.
	 *  \param image Input mask belonging to the input image.
	 *  \return Image vector with masked pixels.
	 */
	static std::vector<cv::Vec3d> maskedPixels(const cv::Mat_<cv::Vec3d>& image, const cv::Mat_<unsigned char>& mask);
	/** Extract unmasked pixels from an image.
	 *  \param image Input image.
	 *  \param image Input mask belonging to the input image.
	 *  \return Image vector with unmasked pixels.
	 */
	static std::vector<cv::Vec3d> unmaskedPixels(const cv::Mat_<cv::Vec3d>& image, const cv::Mat_<unsigned char>& mask);
	/** Extract masked pixels from a superpixel of an image.
	 *  \param image Input image.
	 *  \param superpixel Image coordinates of the superpixel.
	 *  \param image Input mask belonging to the input image.
	 *  \return Image vector with masked pixels.
	 */
	static std::vector<cv::Vec3d> maskedPixels(const cv::Mat_<cv::Vec3d>& image, const std::vector<cv::Point>& superpixel, const cv::Mat_<unsigned char>& mask);
	/** Extract unmasked pixels from a superpixel of an image.
	 *  \param image Input image.
	 *  \param superpixel Image coordinates of the superpixel.
	 *  \param image Input mask belonging to the input image.
	 *  \return Image vector with unmasked pixels.
	 */
	static std::vector<cv::Vec3d> unmaskedPixels(const cv::Mat_<cv::Vec3d>& image, const std::vector<cv::Point>& superpixel, const cv::Mat_<unsigned char>& mask);

public:
	/** Mask pixels with a color component greater or equal a given threshold.
	 *  \param image Input image.
	 *  \param image Input mask belonging to the input image.
	 *  \param threshold Threshold.
	 */
	static void maskSaturatedPixels(const cv::Mat_<cv::Vec3d>& image, cv::Mat_<unsigned char>& mask, double threshold);
	/** Mask pixels with a color component smaller than a given threshold.
	 *  \param image Input image.
	 *  \param image Input mask belonging to the input image.
	 *  \param threshold Threshold.
	 */
	static void maskMinPixels(const cv::Mat_<cv::Vec3d>& image, cv::Mat_<unsigned char>& mask, double threshold);
	/** Mask pixels with the sum of the color component smaller than a given threshold.
	 *  \param image Input image.
	 *  \param image Input mask belonging to the input image.
	 *  \param threshold Threshold.
	 */
	static void maskDarkPixels(const cv::Mat_<cv::Vec3d>& image, cv::Mat_<unsigned char>& mask, double threshold);
	/** Mask pixels at the borders of the image.
	 *  \param image Input image.
	 *  \param image Input mask belonging to the input image.
	 *  \param threshold Width of the border mask in pixels.
	 */
	static void maskBorderPixels(const cv::Mat_<cv::Vec3d>& image, cv::Mat_<unsigned char>& mask, int width);
};

} // namespace iread

#endif // IREAD_COMMON_MASK_H
