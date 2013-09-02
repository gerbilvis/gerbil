#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include "meanshift_config.h"
#include "mfams.h"
#include "progress_observer.h"
#include <multi_img.h>
#ifdef WITH_SEG_FELZENSZWALB
#include <felzenszwalb.h>
#endif

namespace vole {

using std::vector;

class MeanShift {
public:
	MeanShift(const MeanShiftConfig& config) : config(config) {}

	std::pair<int, int> findKL(const multi_img& input, ProgressObserver *progress = NULL);
	cv::Mat1s execute(const multi_img& input, ProgressObserver *progress = NULL,
					  vector<double> *bandwidths = NULL,
					  const multi_img& spinput = multi_img());

#ifdef WITH_SEG_FELZENSZWALB
	static std::vector<fams_point> prepare_sp_points(const FAMS &fams,
									  const gerbil::felzenszwalb::segmap &map);
	static void cleanup_sp_points(std::vector<fams_point> &points);
	static cv::Mat1s segmentImage(const FAMS &fams, const cv::Mat1i &lookup);
#endif

	// terrible hack superpixel sizes
	std::vector<int> spsizes;

private:
	const MeanShiftConfig &config;

};

}

#endif
