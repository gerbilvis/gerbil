/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "meanshift_shell.h"
#include "meanshift.h"

#include <multi_img.h>
#include <stopwatch.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string.h>
#include <labeling.h>

using namespace boost::program_options;

namespace vole {

MeanShiftShell::MeanShiftShell()
 : Command(
		"meanshift",
		config,
		"Johannes Jordan",
		"johannes.jordan@informatik.uni-erlangen.de")
{}

MeanShiftShell::~MeanShiftShell() {}


int MeanShiftShell::execute() {
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

	MeanShift ms(config);
	if (config.findKL) {
	// find K, L
		std::pair<int, int> ret = ms.findKL(input.first);
		config.K = ret.first; config.L = ret.second;
		std::cout << "Found K = " << config.K
		          << "\tL = " << config.L << std::endl;
		return 0;
	}

#ifdef WITH_SEG_FELZENSZWALB2
	// HACK
	//if (config.starting == SUPERPIXEL)
	//	config.pruneMinN = 1;
#endif

	cv::Mat1s labels_mask;
	if (config.sp_original) {
		labels_mask = ms.execute(input.first, NULL, NULL, input.second);
	} else {
		labels_mask = ms.execute(input.first, NULL, NULL, input.first);
	}
	if (labels_mask.empty())
		return 0;

	double mi, ma;
	cv::minMaxLoc(labels_mask, &mi, &ma);
	std::cerr << "min: " << mi << " \tmax: " << ma << std::endl;
	Labeling labels = labels_mask;
	labels.yellowcursor = false;
	labels.shuffle = true;

	// write out beautifully colored label image
	std::string output_name = config.output_directory + "/"
							  + config.output_prefix + "segmentation_rgb.png";
	cv::imwrite(output_name, labels.bgr());

	return 0;
}

std::map<std::string, boost::any> MeanShiftShell::execute(std::map<std::string, boost::any> &input, ProgressObserver *progress) {
	// XXX: for now, gradient/rescale is expected to be done by caller

	boost::shared_ptr<multi_img> inputimg = boost::any_cast<boost::shared_ptr<multi_img> >(input["multi_img"]);

	MeanShift ms(config);
	std::map<std::string, boost::any> output;
	if (config.findKL) {
	// find K, L
		std::pair<int, int> ret = ms.findKL(*inputimg);
		config.K = ret.first; config.L = ret.second;
		std::cout << "Found K = " << config.K
				  << "\tL = " << config.L << std::endl;

		output["findKL.K"] = ret.first;
		output["findKL.L"] = ret.second;
	} else {
		boost::shared_ptr<cv::Mat1s> labels_mask(new cv::Mat1s(ms.execute(*inputimg, progress)));
		output["labels"] = labels_mask;
	}

	return output;
}


void MeanShiftShell::printShortHelp() const {
	std::cout << "Fast adaptive mean shift segmentation by Georgescu" << std::endl;
}


void MeanShiftShell::printHelp() const {
	std::cout << "Fast adaptive mean shift segmentation by Georgescu" << std::endl;
	std::cout << std::endl;
	std::cout << "Please read \"Georgescu et. al: Mean Shift Based Clustering in High\n"
	             "Dimensions: A Texture Classification Example\"";
	std::cout << std::endl;
}

}

