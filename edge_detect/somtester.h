#ifndef SOMTESTER_H
#define SOMTESTER_H

#include "self_organizing_map.h"
#include <multi_img.h>
#include <cv.h>
#include <vector>

class SOMTester
{
public:
    SOMTester(const SOM &map, const multi_img &image, const EdgeDetectionConfig &conf);

	/**
	* Generates the 2D grayscale representation of the multispectral image
	* using the 1D SOM network and the multispectral image by finding
	* the best matching neuron to each pixel vector of the multispectral image.
	*
	* @return Rank image which is the result of the dimensionality reduction
	*/
	cv::Mat generateRankImage();

	/**
	* Generates the 2D grayscale representation of the multispectral image
	* using the 2D SOM network and the multispectral image by finding
	* the best matching neuron to each pixel vector of the multispectral image.
	* Additionally involves linearization of the two dimensional SOM
	* using an appropriate hilbert or peano curve.
	*/
	cv::Mat generateRankImage(cv::Mat_<unsigned int> &rank);

	/**
	* Applies the canny edge detector to the rank/grayscale image.
	* Hysteresis parameter control the edge detection result.
	*
	* @param	h1 lower hysteresis threshold parameter
	* @param	h2 upper hysteresis threshold parameter
	* @return	Edge image of type cv::Mat
	*/
	cv::Mat generateEdgeImage(double h1, double h2);

	/**
	*
	* Compute directional distance images using a fake Sobel operator
	*
	*	@param dx	Horizontal distance map
	*	@param dx	Vertical distance map
	*	@param mode	0: Sobel fake, 1: Scharr fake
	*
	*/
	void getEdge(cv::Mat1d &dx, cv::Mat1d &dy, int mode);

private:
	const SOM &som;
	const multi_img &image;
	const EdgeDetectionConfig &config;
	std::vector<std::vector<cv::Point> > lookup;

	cv::Mat1f rankmap;
	cv::Mat edgemap;
};

#endif // SOMTESTER_H
