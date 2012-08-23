#include "command_seg_felzenszwalb.h"
#include "fh_segmentation.h"

#include <iostream>
#include <cv.h>
#include <highgui.h>

using namespace boost::program_options;

// if we want an enum for configuration, we need to call this
// ENUM_MAGIC(boundcheck);

namespace vole {

CommandSegFelzenszwalb::CommandSegFelzenszwalb()
 : Command(
		"fhseg",
		config,
		"Christian Riess",
		"christian.riess@informatik.uni-erlangen.de")
{
}

CommandSegFelzenszwalb::~CommandSegFelzenszwalb() {
}

int CommandSegFelzenszwalb::execute() {
	std::cout << "Felzenszwalb/Huttenlocher Segmentation" << std::endl;

	if (config.input_file.length() < 1) {
		std::cout << "No input image given!" << std::endl;
		printShortHelp();
		return 1;
	}

	cv::Mat_<cv::Vec3b> img = cv::imread(config.input_file);
	// check whether the image has been successfully loaded
    if (img.empty()) {
        std::cout << "error loading image " << config.input_file << std::endl;
        return -1;
    }

	fhSegmentation fh(config);
	cv::Mat_<cv::Vec3b> segmented = fh.segment(img);
	// FIXME write output image
	// FIXME directory ist directory + image
	if ((config.output_directory.length() > 0) && (config.output_directory != "/tmp/")) {
		cv::imwrite(config.output_directory, segmented);
	} else {
		cv::imwrite("/tmp/fhsegmented.png", segmented);
	}
	
	return 0;
}


void CommandSegFelzenszwalb::printShortHelp() const {
	std::cout << "Superpixel segmentation by Felzenszwalb and Huttenlocher" << std::endl;
}


void CommandSegFelzenszwalb::printHelp() const {
	std::cout << "Superpixel segmentation according to Felzenszwalb and Huttenlocher" << std::endl;
	std::cout << std::endl;
	std::cout << "please read \"Felzenszwalb, Huttenlocher: Efficient Graph-Based Image Segmentation" << std::endl;
}

}

