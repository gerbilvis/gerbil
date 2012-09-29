/**
 *  \file   color_space.h
 *  \brief  A class for colorspace conversions.
 *  \author Michael Bleier
 *  \date   2010
 */

#ifndef IREAD_COLOR_SPACE_H
#define IREAD_COLOR_SPACE_H

#include <opencv2/core/core.hpp>
#include "illum.h"

namespace iread {

/** The ColorSpace class provides helper functions for colorspace conversions */
class ColorSpace
{
public:
	static cv::Vec3d calculateChromaticities(cv::Vec3d color);
	static illumestimators::Illum calculateChromaticities(illumestimators::Illum color);
	static cv::Mat_<cv::Vec3d> colorCorrectImage(const cv::Mat_<cv::Vec3d>& image, const cv::Vec3d& estimate);

public:
	static cv::Mat_<cv::Vec3d> srgb2bgr(const unsigned char* input, int rows, int cols);
	static cv::Mat_<cv::Vec3d> rgb2bgr(const unsigned char* input, int rows, int cols);

	static cv::Mat_<cv::Vec3d> sbgr2bgr(const cv::Mat_<cv::Vec3b>& input);
	static cv::Mat_<cv::Vec3d> bgr2bgr(const cv::Mat_<cv::Vec3b>& input);

	static cv::Vec3d rgb2bgr(const cv::Vec3d& input);
	static cv::Vec3d bgr2rgb(const cv::Vec3d& input);

	static cv::Mat_<cv::Vec3d> sbgr2bgr(const cv::Mat_<cv::Vec3d>& input);
	static cv::Mat_<cv::Vec3d> bgr2sbgr(const cv::Mat_<cv::Vec3d>& input);
};

} // namespace iread

#endif // IREAD_COLOR_SPACE_H
