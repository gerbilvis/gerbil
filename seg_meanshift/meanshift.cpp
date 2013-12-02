/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "meanshift.h"
#include "mfams.h"
#include <multi_img.h>

#ifdef WITH_SEG_FELZENSZWALB
#include <felzenszwalb.h>
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

cv::Mat1s MeanShift::execute(const multi_img& input, ProgressObserver *progress,
							 vector<double> *bandwidths,
							 const multi_img& spinput) {
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

	// HACK it's a shame
	cfams.spsizes = spsizes;

	cfams.ImportPoints(input);

#ifdef WITH_SEG_FELZENSZWALB
	// superpixel setup
	cv::Mat1i sp_translate;
	gerbil::felzenszwalb::segmap sp_map;
	std::vector<fams_point> sp_points; // initialize in right scope!
	if (config.starting == SUPERPIXEL) {
		std::pair<cv::Mat1i, gerbil::felzenszwalb::segmap> result =
			 gerbil::felzenszwalb::segment_image(spinput, config.superpixel);
		sp_translate = result.first;
		std::swap(sp_map, result.second);

		// note: remove output afterwards
		std::cout << "SP: " << sp_map.size() << " segments" << std::endl;
		vole::Labeling output;
		output.yellowcursor = false;
		output.shuffle = true;
		output.read(result.first, false);
		std::string output_name = config.output_directory + "/"
								  + config.output_prefix + "superpixels.png";
		cv::imwrite(output_name, output.bgr());
	}
#endif

	// prepare MS run (adaptive bandwidths)
	bool success = cfams.PrepareFAMS(bandwidths);
	if (!success)
		return cv::Mat1s();

	// define starting points
	/* done after preparation such that superpixel can rely on bandwidths */
	switch (config.starting) {
	case JUMP:
		cfams.SelectMsPoints(0., config.jump);
		break;
	case PERCENT:
		cfams.SelectMsPoints(config.percent, 1);
		break;
#ifdef WITH_SEG_FELZENSZWALB
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
#ifdef WITH_SEG_FELZENSZWALB
/*	if (config.starting == SUPERPIXEL) {
		cfams.DbgSavePoints(config.output_directory + "/sp-points-img",
							sp_points, input.meta);
	}*/
	cleanup_sp_points(sp_points);
#endif
	if (!success)
		return cv::Mat1s();

	// postprocess: prune modes
	cfams.PruneModes();

	if (!config.batch) {
		// save the data
		cfams.SaveModeImg(config.output_directory + "/"
						  + config.output_prefix + "modes", input.meta);
		//cfams.SaveModes(config.output_directory + "/");
		// save pruned modes
		cfams.SavePrunedModeImg(config.output_directory + "/"
						  + config.output_prefix + "prunedmodes", input.meta);
		//cfams.SavePrunedModes(config.output_directory + "/");
		//cfams.SaveMymodes(config.output_directory + "/");
	}

	// return image which holds segment indices of each pixel
	if (config.starting == ALL) {
		return cfams.segmentImage();
#ifdef WITH_SEG_FELZENSZWALB
	} else if (config.starting == SUPERPIXEL) {
		return segmentImage(cfams, sp_translate);
#endif
	} else {
		std::cerr << "Note: As mean shift is not run on all input points, no "
				"output images were created." << std::endl;
		return cv::Mat1s();
	}
}

#ifdef WITH_SEG_FELZENSZWALB
std::vector<fams_point> MeanShift::prepare_sp_points(const FAMS &fams,
								  const gerbil::felzenszwalb::segmap &map)
{
	int D = fams.d_;
	const fams_point *points = fams.getPoints();
	std::vector<fams_point> ret;

	/* while superpixel vectors are averaged, the initial bandwidth is the
	   maximum bandwidth that any individual superpixel member would obtain.
	*/

	std::vector<int> accum(D);
	gerbil::felzenszwalb::segmap::const_iterator mit = map.begin();
	for (; mit != map.end(); ++mit) {
		// initialize new point with zero
		fams_point p;
		p.data_ = new unsigned short[D];
		p.window_ = 0;
		p.weightdp2_ = 0.;

		int N = (int)mit->size();

		// sum up all superpixel members
		std::fill_n(accum.begin(), D, 0);
		for (int i = 0; i < N; ++i) {
			int coord = (*mit)[i];
			for (int d = 0; d < D; ++d)
				accum[d] += points[coord].data_[d];
			p.window_ = std::max(p.window_, points[coord].window_);
			p.weightdp2_ += points[coord].weightdp2_;
		}

		// divide by N to obtain average
		for (int d = 0; d < D; ++d)
			p.data_[d] = accum[d] / N;
		p.weightdp2_ /= (double)N;

		// add to point set
		ret.push_back(p);

		// HACK  tell mfams superpixel size
		fams.spsizes.push_back(N);
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
