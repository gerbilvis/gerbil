/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "meanshift_som.h"
#include "meanshift.h"

#ifdef WITH_EDGE_DETECT
#include <self_organizing_map.h>
#include <som_trainer.h>
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

MeanShiftSOM::MeanShiftSOM()
 : Command(
		"meanshiftsom",
		config,
		"Johannes Jordan",
		"johannes.jordan@informatik.uni-erlangen.de")
{}

MeanShiftSOM::~MeanShiftSOM() {}


int MeanShiftSOM::execute() {
#ifdef WITH_EDGE_DETECT

	multi_img::ptr input = ImgInput(config.input).execute();
	if (input->empty())
		return -1;

	// rebuild before stopwatch for fair comparison
	input->rebuildPixels(false);

	Stopwatch watch("Total time");

	// SOM setup
	if (config.som.hack3d) {
		assert(config.som.height == 1);
		config.som.height = config.som.width * config.som.width;
	}

	SOM som(config.som, input->size());
	std::cout << "# Generated SOM " << config.som.width
			  << "x" << config.som.height << " with dimension "
			  << input->size() << std::endl;

	{
		SOMTrainer trainer(som, *input, config.som);

		std::cout << "# SOM Trainer starts to feed the network using "
				  << config.som.maxIter << " iterations..." << std::endl;
		vole::Stopwatch watch("Training");
		trainer.feedNetwork();
	}

	// create meanshift input
	multi_img msinput = som.export_2d();

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

	cv::Mat1s labels_ms = ms.execute(msinput);
	if (labels_ms.empty())
		return 0;

	double mi, ma;
	cv::minMaxLoc(labels_ms, &mi, &ma);
	std::cerr << "min: " << mi << " \tmax: " << ma << std::endl;

	// translate results back to original image domain
	cv::Mat1s labels_mask(input->height, input->width);
	cv::Mat1s::iterator it = labels_mask.begin();
	for (unsigned int i = 0; it != labels_mask.end(); ++i, ++it) {
		cv::Point n = som.identifyWinnerNeuron(input->atIndex(i));
		*it = labels_ms(n);
	}

	// DBG: write out input to FAMS
	std::string output_name = config.output_directory + "/"
							  + config.output_prefix + "-som";
	msinput.write_out(output_name);

	// write out beautifully colored label image
	Labeling labels = labels_mask;
	labels.yellowcursor = false;
	labels.shuffle = true;
	output_name = config.output_directory + "/"
				  + config.output_prefix + "segmentation_rgb_som.png";
	cv::imwrite(output_name, labels.bgr());

	return 0;
#else
	std::cerr << "FATAL: Edge detection (SOM provider) was not built-in!"
			  << std::endl;
	return 1;
#endif
}

void MeanShiftSOM::printShortHelp() const {
	std::cout << "Fast adaptive mean shift segmentation on SOM" << std::endl;
}


void MeanShiftSOM::printHelp() const {
	std::cout << "Fast adaptive mean shift segmentation on SOM." << std::endl;
	std::cout << std::endl;
	std::cout << "Learns self-organizing map as a pre-processing step.\n";
	std::cout << "Please read \"Georgescu et. al: Mean Shift Based Clustering in High\n"
	             "Dimensions: A Texture Classification Example\"";
	std::cout << std::endl;
}

}

