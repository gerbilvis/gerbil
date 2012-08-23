#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include "meanshift_config.h"
#include "progress_observer.h"
#include <multi_img.h>

namespace vole {

using std::vector;

class MeanShift {
public:
	MeanShift(const MeanShiftConfig& config) : config(config) {}

	std::pair<int, int> findKL(const multi_img& input, ProgressObserver *progress = NULL);
	cv::Mat1s execute(const multi_img& input, ProgressObserver *progress = NULL, vector<double> *bandwidths = NULL);

private:
	const MeanShiftConfig &config;

};

}

#endif
