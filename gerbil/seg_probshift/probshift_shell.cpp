/*	
	Copyright(c) 2011 Daniel Danner,
	Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "probshift_shell.h"
#include "probshift.h"

#include <cv.h>
#include <highgui.h>
#include <multi_img.h>
#include <iostream>
#include <string.h>
#include <labeling.h>

using namespace boost::program_options;

namespace vole {

ProbShiftShell::ProbShiftShell()
 : Command(
		"probshift",
		config,
		"Daniel Danner",
		"daniel@danner.de")
{}

ProbShiftShell::~ProbShiftShell() {}


int ProbShiftShell::execute() {
	multi_img::ptr input = ImgInput(config.inputconfig).execute();

	if (input->empty())
		return -1;

	// make sure the value range is used
	input->data_stretch();
	input->rebuildPixels();

	ProbShift ps(config);

	cv::Mat1s labels_mask = ps.execute(*input, "blackjack");
	Labeling labels = labels_mask;
	labels.yellowcursor = false;

	/// write out beautifully colored label image
	std::string output_name = config.output_directory + "/" + basename(config.inputconfig.file.c_str()) + "_output.png";
	cv::imwrite(output_name, labels.bgr());

	return 0;
}

std::map<std::string, boost::any> ProbShiftShell::execute(std::map<std::string, boost::any> &input, ProgressObserver *progress) {
	// XXX: for now, gradient/rescale is expected to be done by caller

	boost::shared_ptr<multi_img> inputimg = boost::any_cast<boost::shared_ptr<multi_img> >(input["multi_img"]);

	ProbShift ps(config);
	std::map<std::string, boost::any> output;
	boost::shared_ptr<cv::Mat1s> labels_mask(new cv::Mat1s(ps.execute(*inputimg, "probshift", progress)));
	output["labels"] = labels_mask;

	return output;
}


void ProbShiftShell::printShortHelp() const {
	std::cout << "Probabilistic Shift Segmentation by Shetty and Ahuja" << std::endl;
}


void ProbShiftShell::printHelp() const {
	std::cout << "Probabilistic Shift Segmentation by Shetty and Ahuja" << std::endl;
	std::cout << std::endl;
	std::cout << "Please read \"Shetty, Ahuja: Supervised and Unsupervised Clustering with Probabilistic Shift\"";
	std::cout << std::endl;
}

}

