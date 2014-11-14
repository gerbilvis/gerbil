/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "meanshift_som.h"
#include "meanshift.h"

#ifdef WITH_SOM
#include <som_config.h>
#include <gensom.h>
#include <som_cache.h>
#endif
#include <labeling.h>

#include <multi_img.h>
#include <stopwatch.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string.h>
#include <tbb/parallel_for.h>
#include <boost/make_shared.hpp>


using namespace boost::program_options;

namespace seg_meanshift {

MeanShiftSOM::MeanShiftSOM()
 : Command(
		"meanshiftsom",
		config,
		"Johannes Jordan",
		"johannes.jordan@informatik.uni-erlangen.de")
{}

MeanShiftSOM::~MeanShiftSOM() {}

int MeanShiftSOM::execute() {
#ifdef WITH_SOM

	multi_img::ptr input = imginput::ImgInput(config.input).execute();
	if (input->empty())
		return -1;

	// rebuild before stopwatch for fair comparison
	input->rebuildPixels(false);

	Stopwatch watch("Total time");

	Result res = execute(input);
	if (res.labels->empty())
		return 0;

	res.printModes();

	// write out beautifully colored label image
	Labeling labels = *res.labels;
	labels.yellowcursor = false;
	labels.shuffle = true;
	std::string output_name = config.output_directory + "/"
				  + config.output_prefix
				  + "-segmentation_rgb_som.png";
	cv::imwrite(output_name, labels.bgr());

	return 0;
#else
	std::cerr << "FATAL: SOM module was not built-in!"
			  << std::endl;
	return 1;
#endif
}

MeanShiftSOM::Result MeanShiftSOM::execute(multi_img::ptr input)
{
	// SOM setup
	boost::shared_ptr<som::GenSOM> som(som::GenSOM::create(config.som, *input));

	// build lookup table
	boost::shared_ptr<som::SOMClosestN>
			mapping(new som::SOMClosestN(*som, *input, 1));

	// create meanshift input
	multi_img msinput = som->img(input->meta,
								 multi_img_base::Range(
									 input->minval, input->maxval));
/*	if (config.sp_withGrad) {	TODO cleanup
		msinput.apply_logarithm();
		msinput = msinput.spec_gradient();
	}*/

	/* these are flat arrays, but we still use 2d index conversion from the SOM,
	 * because the layout of msinput is based on that! and it is not just a
	 * flattening. */
	std::vector<double> weights(som->size2D().area(), 0.f);
	std::vector<int> spsizes(som->size2D().area(), 0.f);
	{
		Stopwatch watch("Influence-based weight calculation");
		for (int y = 0; y < input->height; ++y) {
			for (int x = 0; x < input->width; ++x) {
				som::SOMClosestN::resultAccess answer =
						mapping->closestN(cv::Point(x, y));
				cv::Point pos = som->getCoord2D(answer.first->index);

				weights[pos.y * msinput.width + pos.x]++;
				spsizes[pos.y * msinput.width + pos.x]++;
			}
		}
	}

	// arrange weights around their mean
	cv::Mat1d wmat(weights);
	double wmean = cv::mean(wmat)[0];
	wmat /= wmean;

	// execute mean shift
	//config.pruneMinN = 1; -> not needed with influence associations
	//config.batch = true;
	MeanShift ms(config);

	// HACK tell SOM neuron influence sizes
	ms.spsizes.swap(spsizes);

	assert(!config.findKL);

	MeanShift::Result msres = ms.execute(msinput, 0,
									   (config.sp_weight > 0 ? &weights : 0));
	if (msres.labels->empty())
		return Result();

	Result ret_out;
	ret_out.modes = msres.modes;
	ret_out.som = som;
	ret_out.lookup = mapping;

	// translate results back to original image domain
	ret_out.labels->create(input->height, input->width);
	{
		Stopwatch watch("Label Image Generation");
		for (int y = 0; y < input->height; ++y) {
			for (int x = 0; x < input->width; ++x) {
				som::SOMClosestN::resultAccess answer =
						mapping->closestN(cv::Point(x, y));
				cv::Point pos = som->getCoord2D(answer.first->index);
				
				// get segement number of 2D coordinate position
				short index = (*msres.labels)(pos) - 1;

				(*ret_out.labels)(y, x) = index;
			}
		}
	}

	std::string output_name;
	if (config.verbosity > 1) {
		// DBG: write out input to FAMS
		output_name = config.output_directory + "/"
					  + config.output_prefix + "-som";
		msinput.write_out(output_name);
		cv::Mat1d wmatXY(som->size2D());
		for (int y = 0; y < wmatXY.rows; ++y) {
			for (int x = 0; x < wmatXY.cols; ++x) {
				wmatXY(y, x) = wmat(y * wmatXY.cols + x, 0);
			}
		}
		output_name = config.output_directory + "/"
					  + config.output_prefix + "-influences.png";
		cv::imwrite(output_name, wmatXY * 127.f);
	}

	return ret_out;
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

