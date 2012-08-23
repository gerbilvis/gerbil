#ifndef VOLE_QUICKSHIFT_H
#define VOLE_QUICKSHIFT_H

#include <iostream>

#include "quickshift_config.h"

#include "cv.h"

namespace vole {

class Quickshift {
public:
	Quickshift(QuickshiftConfig *config, cv::Mat_<cv::Vec3b> img);
	~Quickshift();

	int execute();

	cv::Mat_<cv::Vec3b> getAverageColoredImage();
	cv::Mat_<cv::Vec3b> getRandomlyColoredImage();

	std::map<int, std::vector<std::pair<cv::Point, cv::Vec3b> > > &getSegments();

	QuickshiftConfig *config;

private:
	std::map<int, std::vector<std::pair<cv::Point, cv::Vec3b> > > segments;
	cv::Mat_<cv::Vec3b> img;
	cv::Mat_<cv::Vec3b> averageColored;
	cv::Mat_<cv::Vec3b> randomlyColored;

};

}

#endif // VOLE_QUICKSHIFT_H
