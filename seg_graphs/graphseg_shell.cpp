/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "graphseg_shell.h"
#include "graphseg.h"

#include <multi_img.h>
#include <labeling.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

namespace vole {

GraphSegShell::GraphSegShell()
 : Command(
		"graphseg",
		config,
		"Johannes Jordan",
		"johannes.jordan@informatik.uni-erlangen.de")
{}

int GraphSegShell::execute() {
	multi_img input(config.input_file);
	if (input.empty())
		return -1;

	// Read the seed image
	cv::Mat1b seeds;
	seeds = cv::imread(config.seed_file, CV_LOAD_IMAGE_GRAYSCALE);
	if (seeds.empty()) {
		std::cerr << "ERROR: Could not load seed file: "
		          << config.seed_file << std::endl;
		return -1;
	}

	GraphSeg seg(config);
	cv::Mat1b proba_map;
	cv::Mat1b result = seg.execute(input, seeds, &proba_map);
	if (result.empty())
		return -1;
	
	vole::Labeling output;
	output.yellowcursor = false;
	output.setLabels(result);

	std::string output_name = config.output_file;
	cv::imwrite(output_name, output.grayscale());
	if (!proba_map.empty())
		cv::imwrite(config.output_file + "_proba.png", proba_map);
	return 0;
}


void GraphSegShell::printShortHelp() const {
	std::cout << "Graph Cut / Power Watershed segmentation by Grady" << std::endl;
}


void GraphSegShell::printHelp() const {
	std::cout << "Graph Cut / Power Watershed segmentation by Grady" << std::endl;
	std::cout << std::endl;
	std::cout << "Please read \"Couprie et. al: Power Watersheds: A new image segmentation\n"
	             "framework extending graph cuts, random walker and optimal spanning forest.\"";
	std::cout << std::endl;
}

}

