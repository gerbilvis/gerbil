/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "manipulation_shell.h"
#include <sm_factory.h>
#include <multi_img.h>
#include <iostream>

using namespace boost::program_options;

namespace vole {

ManipulationShell::ManipulationShell()
 : Command(
		"manipulation",
		config,
		"Shuqing Chen",
		"shuqing.chen@studium.uni-erlangen.de")
{}

ManipulationShell::~ManipulationShell() {}


int ManipulationShell::execute() {
	// image cropping and spectra cropping
	multi_img::ptr input = ImgInput(config.m_InputConfig1).execute();
	if (input->empty())
		return -1;

	if (config.task.compare("median")==0) {
		multi_img output(1, 1, input->size());
		for (int i = 0; i < input->size(); ++i) {
			// median of current band in image 2
			std::vector<multi_img::Value> v((*input)[i].begin(), (*input)[i].end());
			std::sort(v.begin(), v.end());
			multi_img::Value median = v[v.size() / 2];
			std::cout << i << ": " << median << std::endl;

			multi_img::Band b(output[i]);
			b.setTo(median);
			output.setBand(i, b);
			output.meta = input->meta;
		}
		std::string output_name = config.m_strOutputFilename;
		output.write_out(output_name);
	}

	if (!(config.task.compare("divide")==0 ||
		config.task.compare("compare")==0))
		return 0;

	multi_img::ptr input2 = ImgInput(config.m_InputConfig2).execute();

	if (input2->empty())
		return -1;

	if (config.task.compare("compare") == 0) {
		input->rebuildPixels(false);
		input2->rebuildPixels(false);
		multi_img::Pixel reference = (*input2)(0, 0);

		SMConfig distcfg[2];
		distcfg[0].measure = EUCLIDEAN;
		distcfg[1].measure = MOD_SPEC_ANGLE;
		SimilarityMeasure<multi_img::Value> *distfun[2];
		distfun[0] = vole::SMFactory<multi_img::Value>::spawn(distcfg[0]);
		distfun[1] = vole::SMFactory<multi_img::Value>::spawn(distcfg[1]);
		assert(distfun[0] && distfun[1]);

		double distsum[2] = { 0., 0.};
		for (int y = 0; y < input->height; ++y) {
			for (int x = 0; x < input->width; ++x) {
				for (int d = 0; d < 2; ++ d) {
					double dist = distfun[d]->getSimilarity((*input)(y, x), reference);
					distsum[d] += std::abs(dist);
				}
			}
		}
		distsum[0] /= input->height * input->width * 255.;
		distsum[1] /= input->height * input->width;
		std::cout << "Mean Euclidean distance: " << distsum[0] << std::endl;
		std::cout << "Mean Spectral Angle error: " << distsum[1] << std::endl;
	}

	if (config.task.compare("divide") == 0) {
		assert(input2->width == 1 && input2->height == 1);
		for (int i = 0; i < input->size(); ++i) {
			multi_img::Band b((*input)[i]);
			b /= (*input2)[i](0, 0);
			input->setBand(i, b);
		}
		input->maxval = 1.;
		std::string output_name = config.m_strOutputFilename;
		input->write_out(output_name);
	}

	return 0;
}


void ManipulationShell::printShortHelp() const {
	std::cout << "Shell implementation for module imginput by Shuqing" << std::endl;
}


void ManipulationShell::printHelp() const {
	std::cout << "Shell implementation for module imginput by Shuqing" << std::endl;
	std::cout << std::endl;
	std::cout << "Provide shell implementation to crop sub image and separate it into multi spectrum";
	std::cout << std::endl;
}

}

