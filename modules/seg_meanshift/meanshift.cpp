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
#include <labeling.h>
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdio>

namespace vole {

std::pair<int, int> MeanShift::findKL(const multi_img& input, ProgressObserver *progress)
{
	// load points
	FAMS cfams(config, progress);
	cfams.ImportPoints(input);
	return cfams.FindKL();
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

	FAMS cfams(config, progress);

	cfams.ImportPoints(input);

#ifdef WITH_SEG_FELZENSZWALB2
	// superpixel setup
	cv::Mat1i sp_translate;
	gerbil::felzenszwalb::segmap sp_map;
	std::vector<fams_point> sp_points; // initialize in right scope!
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

		// remove afterwards
		std::cout << "SP: " << sp_map.size() << " segments" << std::endl;
		vole::Labeling output;
		output.yellowcursor = false;
		output.read(result.first, false);
		std::string output_name = config.output_directory + "/superpixels.png";
		cv::imwrite(output_name, output.bgr());
	}
#endif

	// prepare MS run (adaptive bandwidths)
	bool success = cfams.PrepareFAMS(bandwidths);
	if (!success)
		return multi_img::Mask();

	// define starting points
	/* done after preparation such that superpixel can rely on bandwidths */
	switch (config.starting) {
	case JUMP:
		cfams.SelectMsPoints(0., config.jump);
		break;
	case PERCENT:
		cfams.SelectMsPoints(config.percent, 1);
		break;
#ifdef WITH_SEG_FELZENSZWALB2
	case SUPERPIXEL:
		sp_points = prepare_sp_points(cfams, sp_map);
		cfams.ImportMsPoints(sp_points);
		break;
#endif
	default:
		cfams.SelectMsPoints(0., 1);
	}

	// perform mean shift
	success = cfams.FinishFAMS();
#ifdef WITH_SEG_FELZENSZWALB2
	cleanup_sp_points(sp_points);
#endif
	if (!success)
		return multi_img::Mask();

	// postprocess: prune modes
	cfams.PruneModes();

	if (!config.batch) {
		// save the data
		cfams.SaveModes(config.output_directory + "/");
		// save pruned modes
		cfams.SavePrunedModes(config.output_directory + "/");
		cfams.SaveMymodes(config.output_directory + "/");
	}

	if (config.starting == ALL) {
		// save image which holds segment indices of each pixel
		return cfams.segmentImage();
#ifdef WITH_SEG_FELZENSZWALB2
	} else if (config.starting == SUPERPIXEL) {
		return segmentImage(cfams, sp_translate);
#endif
	} else {
		std::cerr << "Note: As mean shift is not run on all input points, no "
				"output images were created." << std::endl;
		return cv::Mat1s();
	}
}

#ifdef WITH_SEG_FELZENSZWALB2
std::vector<fams_point> MeanShift::prepare_sp_points(const FAMS &fams,
								  const gerbil::felzenszwalb::segmap &map)
{
	int D = fams.d_;
	const fams_point *points = fams.getPoints();
	std::vector<fams_point> ret;

	gerbil::felzenszwalb::segmap::const_iterator mit = map.begin();
	for (; mit != map.end(); ++mit) {
		fams_point p;
		p.data_ = new unsigned short[D];
		std::fill_n(p.data_, D, 0);
		p.window_ = 0;
		p.weightdp2_ = 0.;
		int N = mit->size();
		for (int i = 0; i < N; ++i) {
			int coord = (*mit)[i];
			for (int d = 0; d < D; ++d)
				p.data_[d] += points[coord].data_[d];
			p.window_ += points[coord].window_;
			p.weightdp2_ += points[coord].weightdp2_;
		}
		for (int d = 0; d < D; ++d)
			p.data_[d] /= N;
		p.window_ /= N;
		p.weightdp2_ /= (double)N;
		ret.push_back(p);
	}
	return ret;
}

cv::Mat1s MeanShift::segmentImage(const FAMS &fams, const cv::Mat1i &lookup)
{
	const int *modes = fams.getModePerPixel();
	cv::Mat1s ret(fams.h_, fams.w_);

	cv::Mat1i::const_iterator itl = lookup.begin();
	cv::Mat1s::iterator itr = ret.begin();
	for (; itr != ret.end(); ++itl, ++itr) {
		// keep clear of zero
		*itr = modes[*itl] + 1;
	}

	return ret;
}

void MeanShift::cleanup_sp_points(std::vector<fams_point> &points)
{
	for (int i = 0; i < points.size(); ++i)
		delete[] points[i].data_;
}

#endif

} // namespace
