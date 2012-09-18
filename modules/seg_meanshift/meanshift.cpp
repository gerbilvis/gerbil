/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "meanshift.h"
#include "mfams.h"
#include <multi_img.h>

#ifdef WITH_SEG_FELZENSZWALB2
#include <felzenszwalb.h>
#include <sm_factory.h>
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
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

#ifdef WITH_SEG_FELZENSZWALB2
	// superpixel setup
	cv::Mat1i sp_translate;
	gerbil::felzenszwalb::segmap sp_map;
	if (config.starting == SUPERPIXEL) {
		vole::SimilarityMeasure<multi_img::Value> *distfun;
		distfun = vole::SMFactory<multi_img::Value>::spawn
				(config.superpixel.similarity);
		assert(distfun);
		std::pair<cv::Mat1i, gerbil::felzenszwalb::segmap> result =
			 gerbil::felzenszwalb::segment_image(input, distfun,
							   config.superpixel.c, config.superpixel.min_size);
		sp_translate = result.first;
		std::swap(sp_map, result.second);
	}
	std::cout << "SP: " << sp_map.size() << " segments" << std::endl;
#endif

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
#ifdef WITH_SEG_FELZENSZWALB2
	case SUPERPIXEL:
		break;
#endif
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
#ifdef WITH_SEG_FELZENSZWALB2
	} else if (config.starting == SUPERPIXEL) {
		return cv::Mat1s();
#endif
	} else {
		std::cerr << "Note: As mean shift is not run on all input points, no "
				"output images were created." << std::endl;
		return cv::Mat1s();
	}
}

} // namespace
