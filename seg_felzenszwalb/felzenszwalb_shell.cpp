/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "felzenszwalb_shell.h"
#include "felzenszwalb.h"

#include <sm_factory.h>
#include <labeling.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

namespace gerbil {

FelzenszwalbShell::FelzenszwalbShell()
 : vole::Command(
		"felzenszwalb",
		config,
		"Johannes Jordan",
		"johannes.jordan@informatik.uni-erlangen.de")
{}

int FelzenszwalbShell::execute() {
	multi_img input(config.input_file);
	if (input.empty())
		return -1;

	/* TODO: optional smoothing, and/or using imginput */

	input.rebuildPixels(false);
	std::pair<cv::Mat1i, felzenszwalb::segmap> result =
		 felzenszwalb::segment_image(input, config);

	if (config.verbosity > 0) {	// statistical output
		const felzenszwalb::segmap &segmap = result.second;
		cv::Mat1i sizes((int)segmap.size(), 1);
		cv::Mat1i::iterator sit = sizes.begin();
		felzenszwalb::segmap::const_iterator mit = segmap.begin();
		for (; mit != segmap.end(); ++sit, ++mit)
			*sit = (int)mit->size();
		cv::Scalar mean, stddev;
		cv::meanStdDev(sizes, mean, stddev);
		std::cout << "Found " << result.second.size() << " segments"
				  << " of avg. size " << mean[0]
				  << " (± " << stddev[0] << ")." << std::endl;
	}

	vole::Labeling output;
	output.yellowcursor = false;
	output.shuffle = true;
	output.read(result.first, false); // TODO: we get consecutive index now!

	std::string output_name = config.output_file;
	cv::imwrite(output_name, output.bgr());
	return 0;
}


void FelzenszwalbShell::printShortHelp() const {
	std::cout << "Superpixel Segmentation by Felzenszwalb, Huttenlocher"
				 " on multispectral images" << std::endl;
}


void FelzenszwalbShell::printHelp() const {
	std::cout << "Superpixel Segmentation by Felzenszwalb, Huttenlocher"
				 " on multispectral images" << std::endl;
	std::cout << std::endl;
	std::cout << "Please refer to Felzenszwalb, Huttenlocher: Efficient Graph-Based Image\n"
				 "Segmentation. International Journal of Computer Vision." << std::endl;
	std::cout << std::endl;
}

} // namespace

