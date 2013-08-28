#ifndef SOMTESTER_H
#define SOMTESTER_H

#include "som.h"
#include <multi_img.h>
#include <vector>

class SOMTester
{
public:
	SOMTester(SOM &map, const multi_img &image);

	/**
	* Compute directional distance images using a fake Sobel operator
	*
	*	@param dx	Horizontal distance map (output)
	*	@param dx	Vertical distance map (output)
	*/
	void getEdge(cv::Mat1d &dx, cv::Mat1d &dy);

//protected:
//	void getEdge3(cv::Mat1d &dx, cv::Mat1d &dy); // hack3d version

private:
	SOM &som;
	const multi_img &image;
//	const vole::EdgeDetectionConfig &config;
//	std::vector<std::vector<cv::Point> > lookup;
};

#endif // SOMTESTER_H
