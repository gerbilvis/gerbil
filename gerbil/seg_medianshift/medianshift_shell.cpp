/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "medianshift_shell.h"
#include "medianshift.h"
#include <multi_img.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string.h>
#include <labeling.h>

using namespace boost::program_options;

namespace vole {

MedianShiftShell::MedianShiftShell()
 : Command(
		"medianshift",
		config,
		"Daniel Danner",
		"daniel@danner.de")
{}

MedianShiftShell::~MedianShiftShell() {}


int MedianShiftShell::execute() {
	multi_img::ptr input = ImgInput(config.inputconfig).execute();

	if (input->empty())
		return -1;

	// make sure the value range is used
	input->data_stretch();
	input->rebuildPixels();

	MedianShift ms(config);

	cv::Mat1s labels_mask = ms.execute(*input);

	Labeling labels = labels_mask;
	labels.yellowcursor = false;

	/// write out beautifully colored label image
	std::string output_name = config.output_directory + "/" + "segmentation_rgb.png";
	cv::imwrite(output_name, labels.bgr());

	return 0;
}

std::map<std::string, boost::any> MedianShiftShell::execute(std::map<std::string, boost::any> &input, ProgressObserver *progress) {
	// XXX: for now, gradient/rescale is expected to be done by caller

	boost::shared_ptr<multi_img> inputimg = boost::any_cast<boost::shared_ptr<multi_img> >(input["multi_img"]);

	MedianShift ms(config);
	boost::shared_ptr<cv::Mat1s> labels_mask(new cv::Mat1s(ms.execute(*inputimg, progress)));

	std::map<std::string, boost::any> output;
	output["labels"] = labels_mask;

	return output;
}

void MedianShiftShell::printShortHelp() const {
	std::cout << "Median Shift Segmentation by Shapira et al." << std::endl;
}


void MedianShiftShell::printHelp() const {
	std::cout << "Median Shift Segmentation by Shapira et al." << std::endl;
	std::cout << std::endl;
	std::cout << "Please read \"Shapira et. al: Mode-Detection via Median-Shift";
	std::cout << std::endl;
}

}

