/*
	Copyright(c) 2013 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "rgb.h"

#ifdef WITH_EDGE_DETECT
#include <som_trainer.h>
#endif

#include <stopwatch.h>
#include <multi_img.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

namespace gerbil {

RGB::RGB()
 : Command(
		"rgb",
		config,
		"Johannes Jordan",
		"johannes.jordan@informatik.uni-erlangen.de")
{}

int RGB::execute()
{
	multi_img src = vole::ImgInput(config.input).execute();
	if (src.empty())
		return 1;
	
	cv::Mat3f bgr;
	
	switch (config.algo) {
	case COLOR_XYZ:
		bgr = src.bgr();
		break;
	case COLOR_PCA:
		bgr = executePCA(src);
		break;
	case COLOR_SOM:
#ifdef WITH_EDGE_DETECT
		bgr = executeSOM(src);
		break;
#else
		std::cerr << "FATAL: SOM functionality missing!" << std::endl;
		return 1;
#endif
	default:
		return 1;
	}
	
	if (bgr.empty())
		return 1;

	if (config.verbosity > 1) {
		cv::imshow("Result", bgr);
		cv::waitKey();
	}
	
	cv::imwrite(config.output_file, bgr*255.);
	return 0;
}

cv::Mat3f RGB::executePCA(const multi_img& src)
{
	multi_img pca3 = src.project(src.pca(3));
//	pca3.data_rescale(0., 1.);
	pca3.data_stretch_single(0., 1.); // TODO: make this configurable

//	bgr = pca3.Mat();
	// green: component 1, red: component 2, blue: component 3
	std::vector<cv::Mat> vec(3);
	vec[0] = pca3[2]; vec[1] = pca3[0]; vec[2] = pca3[1];
	cv::Mat3f bgr;
	cv::merge(vec, bgr);
	return bgr;
}

#ifdef WITH_EDGE_DETECT
cv::Mat3f RGB::executeSOM(const multi_img& img)
{
	img.rebuildPixels(false);
	config.som.hack3d = true;
	config.som.height = config.som.width * config.som.width;

	SOM *som = SOMTrainer::train(config.som, img);
	if (som == NULL)
		return cv::Mat3f();

	if (config.som.output_som) {
		multi_img somimg = som->export_2d();
		somimg.write_out(config.output_file + "_som");
	}

	vole::Stopwatch watch("False Color Image Generation");

	cv::Mat3f bgr(img.height, img.width);
	cv::Mat3f::iterator it = bgr.begin();
	if (config.som_depth < 2) {
		for (unsigned int i = 0; it != bgr.end(); ++i, ++it) {
			cv::Point n = som->identifyWinnerNeuron(img.atIndex(i));
			(*it)[0] = n.x;
			(*it)[1] = n.y / som->getWidth();
			(*it)[2] = n.y % som->getWidth();
		}
		bgr /= config.som.width;
	} else {
		int N = config.som_depth;
		// normalize from sum, then stretch out max. coord to 1
		double factor = 1. / (double)(config.som.width * N);
		for (unsigned int i = 0; it != bgr.end(); ++i, ++it) {
			std::vector<std::pair<double, cv::Point> > coords =
					som->closestN(img.atIndex(i), N);
			cv::Point3d avg;
			for (int i = 0; i < coords.size(); ++i) {
				cv::Point3d c(coords[i].second.x,
							  coords[i].second.y / som->getWidth(),
							  coords[i].second.y % som->getWidth());
				if (i == 0)
					avg = c;
				else
					avg += c; // TODO: weighting
			}
			(*it)[0] = (float)(avg.x * factor);
			(*it)[1] = (float)(avg.y * factor);
			(*it)[2] = (float)(avg.z * factor);
		}
	}
	delete som;
	return bgr;
}
#endif

void RGB::printShortHelp() const {
	std::cout << "RGB image creation (true-color or false-color)" << std::endl;
}


void RGB::printHelp() const {
	std::cout << "RGB image creation (true-color or false-color)" << std::endl;
	std::cout << std::endl;
	std::cout << "XYZ does a true-color image creation using a standard white balancing.\n"
	             "PCA and SOM do false-coloring.\"";
	std::cout << std::endl;
}
}

