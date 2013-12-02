/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "meanshift_sp.h"
#include "meanshift.h"

#ifdef WITH_SEG_FELZENSZWALB
#include <felzenszwalb.h>
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

int MeanShiftSP::execute()
{
#ifdef WITH_SEG_FELZENSZWALB
	multi_img::ptr input, input_grad;
	if (config.sp_withGrad) {
		input = ImgInput(config.input).execute();
		input_grad = multi_img::ptr(new multi_img(*input));
		input_grad->apply_logarithm();
		*input_grad = input_grad->spec_gradient();
	} else {
		input = ImgInput(config.input).execute();
	}
	if (input->empty())
		return -1;

	// rebuild before stopwatch for fair comparison
	input->rebuildPixels(false);
	if (config.sp_withGrad) {
		input_grad->rebuildPixels(false);
	}

	cv::Mat1s labels_mask(execute(input, input_grad));
	if (labels_mask.empty())
		return 0;

	// write out beautifully colored label image
	Labeling labels;
	labels = labels_mask;
	labels.yellowcursor = false;
	labels.shuffle = true;

	std::string output_name = config.output_directory + "/"
				  + config.output_prefix + "segmentation_rgb.png";
	cv::imwrite(output_name, labels.bgr());
	return 0;
#else
	std::cerr << "FATAL: Felzenszwalb superpixel segmentation was not built-in!"
			  << std::endl;
	return 1;
#endif
}

std::map<std::string, boost::any> MeanShiftSP::execute(std::map<std::string, boost::any> &input, ProgressObserver *progress) {
	// XXX: for now, gradient/rescale is expected to be done by caller

	boost::shared_ptr<multi_img> inputimg =
			boost::any_cast<boost::shared_ptr<multi_img> >(input["multi_img"]);
	boost::shared_ptr<multi_img> inputgrad;
	if (config.sp_withGrad) {
		inputgrad =
			boost::any_cast<boost::shared_ptr<multi_img> >(input["multi_grad"]);
	}

	// make sure pixel caches are built
	inputimg->rebuildPixels(true);
	if (config.sp_withGrad)
		inputgrad->rebuildPixels(true);

	boost::shared_ptr<cv::Mat1s> labels_mask(new cv::Mat1s(
												 execute(inputimg, inputgrad)));
	std::map<std::string, boost::any> output;
	output["labels"] = labels_mask;
	return output;
}

cv::Mat1s MeanShiftSP::execute(multi_img::ptr input,  multi_img::ptr input_grad)
{
#ifdef WITH_SEG_FELZENSZWALB
	Stopwatch watch("Total time");
	Labeling labels;
	// superpixel setup
	cv::Mat1i sp_translate;
	gerbil::felzenszwalb::segmap sp_map;

	// run superpixel pre-segmentation
	std::pair<cv::Mat1i, gerbil::felzenszwalb::segmap> result =
		 gerbil::felzenszwalb::segment_image(*input, config.superpixel);
	sp_translate = result.first;
	std::swap(sp_map, result.second);

	std::cout << "SP: " << sp_map.size() << " segments" << std::endl;
	if (config.verbosity > 1) {
		std::string output_name;
		vole::Labeling output;
		output.yellowcursor = false;
		output.shuffle = true;
		output.read(result.first, false);
		output_name = config.output_directory + "/"
					  + config.output_prefix + "superpixels.png";
		cv::imwrite(output_name, output.bgr());
	}

	// create meanshift input
	multi_img::ptr in = (config.sp_withGrad ? input_grad : input);

	int D = in->size();
	multi_img msinput((int)sp_map.size(), 1, D);
	msinput.minval = in->minval;
	msinput.maxval = in->maxval;
	msinput.meta = in->meta;
	vector<double> weights(sp_map.size());
	std::vector<int> spsizes; // HACK
	gerbil::felzenszwalb::segmap::const_iterator mit = sp_map.begin();
	for (int ii = 0; mit != sp_map.end(); ++ii, ++mit) {
		// initialize new pixel with zero
		multi_img::Pixel p(D, 0.f);

		multi_img::Value N = (multi_img::Value)mit->size();

		// sum up all superpixel members
		for (int i = 0; i < N; ++i) {
			const multi_img::Pixel &s = in->atIndex((*mit)[i]);
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

		// HACK superpixel sizes
		spsizes.push_back((int)N);
	}

	// arrange weights around their mean
	cv::Mat1d wmat(weights);
	double wmean = cv::mean(wmat)[0];
	wmat /= wmean;

	// execute mean shift
	// HACK config.pruneMinN = 1;
	//config.batch = true;
	MeanShift ms(config);

	// HACK tell superpixel sizes
	ms.spsizes.swap(spsizes);

	if (config.findKL) {
		// find K, L
		std::pair<int, int> ret = ms.findKL(msinput);
		config.K = ret.first; config.L = ret.second;
		std::cout << "Found K = " << config.K
				  << "\tL = " << config.L << std::endl;
		return cv::Mat1s();
	}

	cv::Mat1s labels_ms = ms.execute(msinput, NULL,
									 (config.sp_weight > 0 ? &weights : NULL));
	if (labels_ms.empty())
		return cv::Mat1s();

	// DBG: write out input to FAMS
	//output_name = config.output_directory + "/"
	//			  + config.output_prefix + "-spimg";
	//	msinput.write_out(output_name);

	// translate results back to original image domain
	cv::Mat1s labels_mask(in->height, in->width);
	cv::Mat1s::iterator itr = labels_mask.begin();
	cv::Mat1i::const_iterator itl = sp_translate.begin();
	for (; itr != labels_mask.end(); ++itl, ++itr) {
		*itr = labels_ms(*itl, 0);
	}
	return labels_mask;
#endif // WITH_SEG_FELZENSWALB
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

