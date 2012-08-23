/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "meanshift.h"
#include "mfams.h"

#include <multi_img.h>
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <cstdio>

namespace vole {

std::pair<int, int> MeanShift::findKL(const multi_img& input, ProgressObserver *progress)
{
	// load points
	FAMS cfams(config.use_LSH);
	cfams.ImportPoints(input);
	cfams.progressObserver = progress;
	return cfams.FindKL(config.Kmin, config.K, config.Kjump, config.L,
	                    config.k, config.bandwidth, config.epsilon);	
}

cv::Mat1s MeanShift::execute(const multi_img& input, ProgressObserver *progress, vector<double> *bandwidths) {
	std::cout << "Mean Shift Segmentation" << std::endl;

	if (config.seed != 0) {
		std::cout << "Using fixed seed " << config.seed << std::endl;
		srand(config.seed);
	} else {
		time_t tt;
		time(&tt);
		srand((unsigned int) tt);
	}

	FAMS cfams(config.use_LSH);
	cfams.progressObserver = progress;

	cfams.ImportPoints(input);

	// actually run MS
	int cancel;
	switch (config.starting) {
	case JUMP:
		cancel = cfams.RunFAMS(config, config.jump,
				  config.bandwidth);
		break;
	case PERCENT:
		cancel = cfams.RunFAMS(config, config.percent,
				  config.bandwidth);
		break;
	default:
		cancel = cfams.RunFAMS(config, config.bandwidth, bandwidths);
	}

	if (cancel) {
		return multi_img::Mask();
	}

	if (!config.batch) {
		// save the data
		cfams.SaveModes(config.output_directory + "/");
		// save pruned modes
		cfams.SavePrunedModes(config.output_directory + "/");
		cfams.SaveMymodes(config.output_directory + "/");
	}

	if (config.starting == ALL) {
		// save image which holds segment indices of each pixel
		return cfams.segmentImage(false);
	} else {
		std::cerr << "Note: As mean shift is not run on all input points, no "
				"output images were created." << std::endl;
		return cv::Mat1s();
	}
}

} // namespace
