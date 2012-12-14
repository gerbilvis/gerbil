/*	
	Copyright(c) 2012 Ralph Muessig	and Johannes Jordan
	<johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

//system includes
#include <iostream>
#include <sstream>
#include <fstream>

//cv includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//vole includes
#include <stopwatch.h>

//som includes
#include "edge_detection.h"
#include "self_organizing_map.h"
#include "som_trainer.h"
#include "som_tester.h"

EdgeDetection::EdgeDetection() 
  : vole::Command("edge_detect", // command name
          config,
		  "Johannes Jordan, Ralph Muessig", // author names
		  "johannes.jordan@cs.fau.de" ) // email
{}

EdgeDetection::EdgeDetection(const vole::EdgeDetectionConfig &cfg)
  : vole::Command("edge_detect", // command name
		  config,
		  "Johannes Jordan, Ralph Muessig", // author names
		  "johannes.jordan@cs.fau.de" ), // email
 		  config(cfg)
{}

SOM* EdgeDetection::train(const multi_img &img)
{
	if (config.som_file.empty()) {
		vole::Stopwatch running_time("Total running time");
		SOM *som = new SOM(config, img.size());
		std::cout << "# Generated SOM " << config.width << "x" << config.height << " with dimension " << img.size() << std::endl;

		SOMTrainer trainer(*som, img, config);

		std::cout << "# SOM Trainer starts to feed the network using "<< config.maxIter << " iterations..." << std::endl;

		vole::Stopwatch watch("Training");
		trainer.feedNetwork();

		return som;
	} else {
		multi_img somimg;
		somimg.minval = img.minval;
		somimg.maxval = img.maxval;
		somimg.read_image(config.som_file);
		if (somimg.empty()) {
			std::cerr << "Could not read image containing the SOM!" << std::endl;
			return NULL;
		}
		if (somimg.width != config.width || somimg.height != config.height
			|| somimg.size() != img.size()) {
			std::cerr << "SOM image has wrong dimensions!" << std::endl;
			return NULL;
		}
		somimg.rebuildPixels(false);
		return new SOM(config, somimg);
	}
}

int EdgeDetection::execute()
{
	assert(!config.prefix_enabled); // input, output file variables set

	multi_img img;
	img.minval = 0.;
	img.maxval = 1.;
	img.read_image(config.input_file);
	if (img.empty())
		return -1;

	/* hack: apply spec.grad. */
	//image needs to be loaded w/ maxval 255
	//img.apply_logarithm();
	//img = img.spec_gradient();

	img.rebuildPixels(false);

	if (config.hack3d) {
		assert(config.height == 1);
		config.height = config.width * config.width;
	}

	SOM *som = train(img);
	if (som == NULL)
		return -1;

	if (config.output_som) {
		multi_img somimg = som->export_2d();
		somimg.write_out(config.output_dir + "/som");
		config.storeConfig((config.output_dir + "/config.txt").c_str());
	}

	if (config.output_rgb && config.hack3d) {
		vole::Stopwatch watch("False Color Image Generation");

		// TODO: to generate more values, use a mix of N winner neurons

		cv::Mat3b sombgr(img.height, img.width);
		cv::Mat3b::iterator it = sombgr.begin();
		if (false) {
			for (unsigned int i = 0; it != sombgr.end(); ++i, ++it) {
				cv::Point n = som->identifyWinnerNeuron(img.atIndex(i));
				(*it)[0] = n.x;
				(*it)[1] = n.y / som->getWidth();
				(*it)[2] = n.y % som->getWidth();
			}
			sombgr *= 255 / config.width;
		} else {
			int N = 10;
			// normalize from sum, then stretch out max. coord to 255
			double factor = 255. / (double)(config.width * N);
			for (unsigned int i = 0; it != sombgr.end(); ++i, ++it) {
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
						avg += c;
				}
				(*it)[0] = uchar(avg.x * factor);
				(*it)[1] = uchar(avg.y * factor);
				(*it)[2] = uchar(avg.z * factor);
			}
		}
		cv::imwrite(config.output_dir + "/rgb.png", sombgr);
		delete som;
		return 0;
	}

	std::cout << "# Generating 2D image using the SOM and the multispectral image..." << std::endl;
	vole::Stopwatch watch("Edge Image Generation");
	SOMTester tester(*som, img, config);

	cv::Mat1d dX, dY;
	tester.getEdge(dX, dY);
	cv::Mat sobelXShow, sobelYShow;

	dX.convertTo(sobelXShow, CV_8UC1, 255.);
	cv::imwrite(config.output_dir + "/dx.png", sobelXShow);

	dY.convertTo(sobelYShow, CV_8UC1, 255.);
	cv::imwrite(config.output_dir + "/dy.png", sobelYShow);
    
	delete som;
	return 0;
}


void EdgeDetection::printShortHelp() const 
{
	std::cout << "Edge detection in multispectral images using SOM." << std::endl;
}

void EdgeDetection::printHelp() const 
{
	std::cout << "Edge detection in multispectral images using SOM." << std::endl;
	std::cout << std::endl;
	std::cout << "Please read \"Jordan, J., Angelopoulou E.: Edge Detection in Multispectral\n"
	             "Images Using the N-Dimensional Self-Organizing Map.\" (ICIP 2011)"
	          << std::endl;
}
