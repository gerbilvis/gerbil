#ifndef PIXEL_OPERATIONS_H
#define PIXEL_OPERATIONS_H

/**
 * \file pixel_operations.h
 * \author Christian Riess
 * \date 2009-10-08
 */


#include "data_traits.h"
#include <vector>
#include <cmath>
#include "cv.h"

/**
 * operations on vectors of pixels
 */
namespace PixelOperations {

	cv::Mat_<cv::Vec3b> getChromaticityImage255(const cv::Mat_<cv::Vec3b> &intensity_img);

	/** create a new vector of pixels with pixel chromaticities computed according
	 * to channel/sum(channels).
	 * \param rgbVec input pixel vector in rgb colors
	 * \param chromVec container for the result; for each pixel in rgbVec and each
	 * colorchannel chan, chan/sum(channels) is computed.
	 */
	template<typename T>
	void getChromaticitiesVector(const std::vector<cv::Vec<T, 3> > &rgbVec, std::vector<cv::Vec3d> &chromVec);

	/** Determines the minimum intensity and the maximum intensity from a vector of
	 * grayscale pixels.
	 * \param pixels vector of grayscale (i.e. one color channel) pixels
	 * \param min the minimum intensity in pixels
	 * \param max the maximum intensity in pixels
	 */
	template<typename T>
	void getMinMax(std::vector<T> &pixels, T &min, T &max);

	/** Returns the sum of each pixel.
	 * \param rgbVec a 3-color channel input vector of pixels
	 * \param sumVec a 1-color channel output vector containing the channel sum of the input pixels
	 */
	template<typename T, typename U>
	void getVectorPixelSum(const std::vector<cv::Vec<T, 3> > &rgbVec, std::vector<U> &sumVec);

	/** Computes channelwise the mean value of the input pixels.
	 * \param rgbVec a 3-color channel input vector of pixels
	 * \param meanRed the mean red channel intensity
	 * \param meanGreen the mean green channel intensity
	 * \param meanBlue the mean blue channel intensity
	 */
	template<typename T>
	void meanPerChannel(const std::vector<cv::Vec<T, 3> > &inputVec, double &meanRed, double &meanGreen, double &meanBlue);

	/** Computes channelwise the standard deviation of the input pixels.
	 * \param rgbVec a 3-color channel input vector of pixels
	 * \param meanRed the standard deviation of the red channel intensities
	 * \param meanGreen the standard deviation of the green channel intensities
	 * \param meanBlue the standard deviation of the blue channel intensities
	 */
	template<typename T>
	void stdDevPerChannel(const std::vector<cv::Vec<T, 3> > &inputVec, double &stdDevRed, double &stdDevGreen, double &stdDevBlue);


	/** for a given set of pixels, this function excludes those for which a
	  * single value in any of the three channels is below a threshold of
	  * min_intensity or above a threshold of max_intensity. During this
	  * process, the pixel values are also hard-scaled to [0;1] by dividing
	  * with 255.
	  * The idea is to take care of noisy values caused by low-quality
	  * cameras. The noise is most influencial at the end or the top of the
	  * range of values, i.e. 0 and 255.
	  * \param input the input as a vector of OpenCV Vec3d objects
	  * \param filtered the output as a vector of OpenCV Vec3d objects. Contains only the remaining pixels.
	  * \param min_intensity the minimum intensity threshold. For a pixel, each color channel's value must be larger than this to get accepted.
	  * \param max_intensity the maximum intensity threshold. For a pixel, each color channel's value must be smaller than this to get accepted.
	  */
	void excludeDarkBrightAndScalePixels(
		std::vector<cv::Vec3d> &input,
		std::vector<cv::Vec3d> &filtered,
		double min_intensity, double max_intensity);

	/** for a given set of pixels, this function excludes those for which a
	  * single value in any of the three channels is below a threshold of
	  * min_intensity or above a threshold of max_intensity. During this
	  * process, the pixel values are also hard-scaled to [0;1] by dividing
	  * with 255.
	  * The idea is to take care of noisy values caused by low-quality
	  * cameras. The noise is most influencial at the end or the top of the
	  * range of values, i.e. 0 and 255.
	  * \param region the input as an OpenCV matrix of OpenCV Vec3b pixels, representing an image region
	  * \param filtered the output as a vector of OpenCV Vec3d objects. Contains only the remaining pixels.
	  * \param min_intensity the minimum intensity threshold. For a pixel, each color channel's value must be larger than this to get accepted.
	  * \param max_intensity the maximum intensity threshold. For a pixel, each color channel's value must be smaller than this to get accepted.
	  */
	void excludeDarkBrightAndScalePixels(
		cv::Mat_<cv::Vec3b> &region,
		std::vector<cv::Vec3d> &filtered,
		double min_intensity, double max_intensity);

	template<typename T>
	void removeDuplicates(
		std::vector<cv::Vec<T, 3> > &pixel_list,
		std::vector<cv::Vec<T, 3> > &unique_pixels);
}

#include "pixel_operations.txx"

#endif // PIXEL_OPERATIONS_H
