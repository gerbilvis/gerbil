/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#define WITH_SEG_FELZENSZWALB2

#include "meanshift_sp.h"
#include "meanshift.h"

#ifdef WITH_SEG_FELZENSZWALB2
#include <felzenszwalb.h>
#include <sm_factory.h>
#include <labeling.h>
#endif

#include <multi_img.h>
#include <stopwatch.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string.h>
#include <labeling.h>

using namespace boost::program_options;

namespace vole {

MeanShiftSP::MeanShiftSP()
 : Command(
		"meanshiftsp",
		config,
		"Johannes Jordan",
		"johannes.jordan@informatik.uni-erlangen.de")
{}

MeanShiftSP::~MeanShiftSP() {}


int MeanShiftSP::execute() {
#ifdef WITH_SEG_FELZENSZWALB2

	std::pair<multi_img, multi_img> input;
	if (config.sp_original) {
		input = ImgInput(config.input).both();
	} else {
		input.first = ImgInput(config.input).execute();
	}
	if (input.first.empty())
		return -1;

	// rebuild before stopwatch for fair comparison
	input.first.rebuildPixels(false);
	if (config.sp_original) {
		input.second.rebuildPixels(false);
	}

	Stopwatch watch("Total time");

	// superpixel setup
	cv::Mat1i sp_translate;
	gerbil::felzenszwalb::segmap sp_map;

	// run superpixel pre-segmentation
	vole::SimilarityMeasure<multi_img::Value> *distfun;
	distfun = vole::SMFactory<multi_img::Value>::spawn
			(config.superpixel.similarity);
	assert(distfun);
	std::pair<cv::Mat1i, gerbil::felzenszwalb::segmap> result =
		 gerbil::felzenszwalb::segment_image(
				 (config.sp_original ? input.second : input.first),
				 distfun, config.superpixel.c, config.superpixel.min_size);
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

	// create meanshift input
	int D = input.first.size();
	multi_img msinput(sp_map.size(), 1, D);
	msinput.minval = input.first.minval;
	msinput.maxval = input.first.maxval;
	msinput.meta = input.first.meta;
	vector<double> weights(sp_map.size());
	gerbil::felzenszwalb::segmap::const_iterator mit = sp_map.begin();
	for (int ii = 0; mit != sp_map.end(); ++ii, ++mit) {
		// initialize new pixel with zero
		multi_img::Pixel p(D, 0.f);

		multi_img::Value N = (multi_img::Value)mit->size();

		// sum up all superpixel members
		for (int i = 0; i < N; ++i) {
			const multi_img::Pixel &s = input.first.atIndex((*mit)[i]);
			for (int d = 0; d < D; ++d)
				p[d] += s[d];
		}
		// divide by N to obtain average
		for (int d = 0; d < D; ++d)
			p[d] /= N;

		// add to ms input
		msinput.setPixel(ii, 0, p);

		// add to weights
		weights[ii] = (double)N; // TODO: sqrt?
	}
	for (int i = 0; i < weights.size(); ++i)
		std::cout << weights[i] << "\t";
	std::cout << std::endl;
	cv::Mat1d wmat(weights);
	double wmean = cv::mean(wmat)[0];
	wmat /= wmean;
	std::cout << std::endl;
	for (int i = 0; i < weights.size(); ++i)
		std::cout << weights[i] << "\t";
	std::cout << std::endl;

	// execute mean shift
	config.pruneMinN = 1;
	//config.batch = true;
	MeanShift ms(config);

	if (config.findKL) {
		// find K, L
		std::pair<int, int> ret = ms.findKL(msinput);
		config.K = ret.first; config.L = ret.second;
		std::cout << "Found K = " << config.K
		          << "\tL = " << config.L << std::endl;
		return 0;
	}

	cv::Mat1s labels_ms = ms.execute(msinput, NULL,
									 (config.sp_weightdp2 ? &weights : NULL));
	if (labels_ms.empty())
		return 0;

	double mi, ma;
	cv::minMaxLoc(labels_ms, &mi, &ma);
	std::cerr << "min: " << mi << " \tmax: " << ma << std::endl;

	// translate results back to original image domain
	cv::Mat1s labels_mask(input.first.height, input.first.width);
	cv::Mat1s::iterator itr = labels_mask.begin();
	cv::Mat1i::const_iterator itl = sp_translate.begin();
	for (; itr != labels_mask.end(); ++itl, ++itr) {
		*itr = labels_ms(*itl, 0);
	}

	// DBG: write out input to FAMS
	output_name = config.output_directory + "/"
				  + config.output_prefix + "-spimg";
	msinput.write_out(output_name);

	// write out beautifully colored label image
	Labeling labels = labels_mask;
	labels.yellowcursor = false;
	labels.shuffle = true;
	output_name = config.output_directory + "/"
				  + config.output_prefix + "segmentation_rgb.png";
	cv::imwrite(output_name, labels.bgr());

	return 0;
#else
	std::cerr << "FATAL: Felzenszwalb superpixel segmentation was not built-in!"
			  << std::endl;
	return 1;
#endif
}

void MeanShiftSP::printShortHelp() const {
	std::cout << "Fast adaptive mean shift segmentation on superpixels" << std::endl;
}


void MeanShiftSP::printHelp() const {
	std::cout << "Fast adaptive mean shift segmentation on superpixels" << std::endl;
	std::cout << std::endl;
	std::cout << "Applies superpixel segmentation as a pre-processing step.\n";
	std::cout << "Please read \"Georgescu et. al: Mean Shift Based Clustering in High\n"
	             "Dimensions: A Texture Classification Example\"";
	std::cout << std::endl;
}

}

