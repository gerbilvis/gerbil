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
#include <algorithm>

namespace seg_meanshift {

KLResult MeanShift::findKL(const multi_img& input, ProgressObserver *po)
{
	// load points
	FAMS cfams(config, po);
	cfams.importPoints(input);
	return cfams.FindKL();
}

MeanShift::Result MeanShift::execute(const multi_img& input,
                                     ProgressObserver *po,
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

	FAMS cfams(config, po);

	// HACK it's a shame
	cfams.spsizes = spsizes;

	cfams.importPoints(input);

#ifdef WITH_SEG_FELZENSZWALB
	// superpixel setup
	cv::Mat1i sp_translate;
	seg_felzenszwalb::segmap sp_map;
	std::vector<FAMS::Point> sp_points; // initialize in right scope!
	if (config.starting == SUPERPIXEL) {
		std::pair<cv::Mat1i, seg_felzenszwalb::segmap> result =
			 seg_felzenszwalb::segment_image(spinput, config.superpixel);
		sp_translate = result.first;
		std::swap(sp_map, result.second);

		// note: remove output afterwards
		std::cout << "SP: " << sp_map.size() << " segments" << std::endl;
		Labeling output;
		output.yellowcursor = false;
		output.shuffle = true;
		output.read(result.first, false);
		std::string output_name = config.output_directory + "/"
								  + config.output_prefix + "-superpixels.png";
		cv::imwrite(output_name, output.bgr());
	}
#endif

	// prepare MS run (adaptive bandwidths)
	bool success = cfams.prepareFAMS(bandwidths);
	if (!success)
		return Result();

	// define starting points
	/* done after preparation such that superpixel can rely on bandwidths */
	switch (config.starting) {
	case JUMP:
		cfams.selectStartPoints(0., config.jump);
		break;
	case PERCENT:
		cfams.selectStartPoints(config.percent, 1);
		break;
#ifdef WITH_SEG_FELZENSZWALB
	case SUPERPIXEL:
		sp_points = prepare_sp_points(cfams, sp_map);
		cfams.importStartPoints(sp_points);
		break;
#endif
	default:
		cfams.selectStartPoints(0., 1);
	}

	// perform mean shift
	success = cfams.finishFAMS();
#ifdef WITH_SEG_FELZENSZWALB
/*	if (config.starting == SUPERPIXEL) {
		cfams.DbgSavePoints(config.output_directory + "/sp-points-img",
							sp_points, input.meta);
	}*/
	cleanup_sp_points(sp_points);
#endif
	if (!success)
		return Result();

	// postprocess: prune modes
	cfams.pruneModes();

	if (config.verbosity > 1) {
		// save the data
		cfams.saveModeImg(config.output_directory + "/"
						  + config.output_prefix + "-modes", 0, input.meta);
		//cfams.SaveModes(config.output_directory + "/modes.txt", 0);
		// save pruned modes
		cfams.saveModeImg(config.output_directory + "/"
						  + config.output_prefix + "-prunedmodes", 2, input.meta);
	}

	std::vector<multi_img::Pixel> modes = cfams.modeVector();

	// return image which holds segment indices of each pixel
	Result ret;
	ret.setModes(cfams.modeVector());
	if (config.starting == ALL) {
		ret.setLabels(cfams.segmentImage());
#ifdef WITH_SEG_FELZENSZWALB
	} else if (config.starting == SUPERPIXEL) {
		ret.setLabels(segmentImageSP(cfams, sp_translate));
#endif
	} else {
		std::cerr << "Note: As mean shift is not run on all input points, no "
				"output images were created." << std::endl;
	}
	return ret;
}

#ifdef WITH_SEG_FELZENSZWALB
std::vector<FAMS::Point> MeanShift::prepare_sp_points(const FAMS &fams,
								  const seg_felzenszwalb::segmap &map)
{
	int D = fams.d_;
	const std::vector<FAMS::Point>& points = fams.getPoints();
	std::vector<FAMS::Point> ret;

	/* while superpixel vectors are averaged, the initial bandwidth is the
	   maximum bandwidth that any individual superpixel member would obtain.
	*/

	std::vector<int> accum(D);
	seg_felzenszwalb::segmap::const_iterator mit;
	for (mit = map.begin(); mit != map.end(); ++mit) {
		// initialize new point with zero
		FAMS::Point p;
		p.data = new std::vector<unsigned short>(D);
		p.window = 0;
		p.weightdp2 = 0.;

		int N = (int)mit->size();

		// sum up all superpixel members
		std::fill_n(accum.begin(), D, 0);
		for (int i = 0; i < N; ++i) {
			int coord = (*mit)[i];
			for (int d = 0; d < D; ++d)
				accum[d] += (*points[coord].data)[d];
			p.window = std::max(p.window, points[coord].window);
			p.weightdp2 += points[coord].weightdp2;
		}

		// divide by N to obtain average
		for (int d = 0; d < D; ++d)
			(*p.data)[d] = accum[d] / N;
		p.weightdp2 /= (double)N;

		// add to point set
		ret.push_back(p);

		// HACK  tell mfams superpixel size
		fams.spsizes.push_back(N);
	}
	return ret;
}

cv::Mat1s MeanShift::segmentImageSP(const FAMS &fams, const cv::Mat1i &lookup)
{
	const std::vector<int> &modes = fams.getModePerPixel();
	cv::Mat1s ret(fams.h_, fams.w_);

	cv::Mat1i::const_iterator itl = lookup.begin();
	cv::Mat1s::iterator itr = ret.begin();
	for (; itr != ret.end(); ++itl, ++itr) {
		// keep clear of zero
		*itr = modes[*itl] + 1;
	}

	return ret;
}

void MeanShift::cleanup_sp_points(std::vector<FAMS::Point> &points)
{
	for (size_t i = 0; i < points.size(); ++i)
		delete points[i].data;
}

#endif

} // namespace
