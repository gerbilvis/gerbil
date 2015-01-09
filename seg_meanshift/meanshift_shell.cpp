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

namespace seg_meanshift {

MeanShiftShell::MeanShiftShell(ProgressObserver *po)
 : Command(
		"meanshift",
		config,
		"Johannes Jordan",
		"johannes.jordan@informatik.uni-erlangen.de",
	   po)
{}

MeanShiftShell::~MeanShiftShell() {}


int MeanShiftShell::execute() {
	multi_img::ptr input, input_grad;
#ifdef WITH_SEG_FELZENSZWALB
	if (config.sp_withGrad) {
		input = imginput::ImgInput(config.input).execute();
		input_grad = multi_img::ptr(new multi_img(*input, true));
		input_grad->apply_logarithm();
		*input_grad = input_grad->spec_gradient();
	} else
#endif
    {
		input = imginput::ImgInput(config.input).execute();
	}
	if (input->empty())
		return -1;

	// rebuild before stopwatch for fair comparison
	input->rebuildPixels(false);
#ifdef WITH_SEG_FELZENSZWALB
	if (config.sp_withGrad) {
		input_grad->rebuildPixels(false);
	}
#endif

	Labeling labels;
	{
		Stopwatch watch("Total time");

		MeanShift ms(config);
		if (config.findKL) {
			// find K, L
			KLResult ret = ms.findKL(
#ifdef WITH_SEG_FELZENSZWALB
						(config.sp_withGrad ? *input_grad : *input));
#else
						*input);
#endif
			diagnoseKLResult(ret);
			std::cout << "Found K = " << config.K
					  << "\tL = " << config.L << std::endl;
			config.K = ret.K; config.L = ret.L;

			return 0;
		}

	#ifdef WITH_SEG_FELZENSZWALB
		// HACK
		//if (config.starting == SUPERPIXEL)
		//	config.pruneMinN = 1;
	#endif

		MeanShift::Result res;
#ifdef WITH_SEG_FELZENSZWALB
		if (config.sp_withGrad) {
			// mean shift on grad, but SP on image
			res = ms.execute(*input_grad, NULL, NULL, *input);
		} else
#endif
		{
			res = ms.execute(*input, NULL, NULL, *input);
		}

		// something went wrong, there must be at least one mode!
		if (res.modes->empty())
			return 1;

		res.printModes();

		if (res.labels->empty())
			return 0;

		labels = *res.labels;
		labels.yellowcursor = false;
		labels.shuffle = true;
	}

	// write out beautifully colored label image
	std::string output_name = config.output_directory + "/"
							  + config.output_prefix + "segmentation_rgb.png";
	cv::imwrite(output_name, labels.bgr());

	return 0;
}

std::map<std::string, boost::any>
MeanShiftShell::execute(std::map<std::string, boost::any> &input,
						ProgressObserver *progress)
{
	// XXX: for now, gradient/rescale is expected to be done by caller

	setProgressObserver(progress);

	boost::shared_ptr<multi_img> inputimg =
	        boost::any_cast<boost::shared_ptr<multi_img> >(input["multi_img"]);
	boost::shared_ptr<multi_img> inputgrad;
#ifdef WITH_SEG_FELZENSZWALB
	if (config.sp_withGrad) {
		inputgrad =
			boost::any_cast<boost::shared_ptr<multi_img> >(input["multi_grad"]);
	}
#endif

	// make sure pixel caches are built
	inputimg->rebuildPixels(true);

#ifdef WITH_SEG_FELZENSZWALB
	if (config.sp_withGrad)
		inputgrad->rebuildPixels(true);
#endif

	MeanShift ms(config);
	std::map<std::string, boost::any> output;
	if (config.findKL) {
		// find K, L
		KLResult res = ms.findKL(
#ifdef WITH_SEG_FELZENSZWALB
				(config.sp_withGrad ? *inputgrad : *inputimg));
#else
				*inputimg);
#endif
		if (res.isGood()) {
				config.K = res.K; config.L = res.L;
				std::cout << "Found K = " << config.K
				<< "\tL = " << config.L << std::endl;
		}
		res.insertInto(output);
		return output;
	} else {
		MeanShift::Result res = ms.execute(
#ifdef WITH_SEG_FELZENSZWALB
				  (config.sp_withGrad ? *inputgrad : *inputimg),
#else
				  *inputimg,
#endif
				   progress, NULL, *inputimg);
		if (res.modes->size() != 0) {
			output["labels"] = res.labels;
			output["modes"] = res.modes;

		}
		return output;
	}
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

