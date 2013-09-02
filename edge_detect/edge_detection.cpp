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

int EdgeDetection::execute()
{
	assert(!config.prefix_enabled); // input, output file variables set

	multi_img img;
	img.minval = 0.;
	img.maxval = 1.;
	img.read_image(config.input_file);
	if (img.empty())
		return -1;

	img.rebuildPixels(false);

	SOM *som = SOMTrainer::train(config, img);
	if (som == NULL)
		return -1;

	if (config.output_som) {
		multi_img somimg = som->export_2d();
		somimg.write_out(config.output_dir + "/som");
		config.storeConfig((config.output_dir + "/config.txt").c_str());
	}

	std::cout << "# Generating 2D image using the SOM and the multispectral image..." << std::endl;
	vole::Stopwatch watch("Edge Image Generation");

	cv::Mat1d dX, dY;
	som->getEdge(img, dX, dY);
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

