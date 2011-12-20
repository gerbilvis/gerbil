//system includes
#include <iostream>
#include <sstream>
#include <fstream>

//cv includes
#include <cv.h>
#include <highgui.h>

//vole includes
#include <stopwatch.h>

//som includes
#include "edge_detection.h"
#include "self_organizing_map.h"
#include "graphsom.h"
#include "som_trainer.h"
#include "somtester.h"
#include "space_filling_curve.h"

EdgeDetection::EdgeDetection() 
  : vole::Command("edge_detect", // command name
          config,
		  "Johannes Jordan, Ralph Muessig", // author names
		  "johannes.jordan@cs.fau.de" ) // email
{}

EdgeDetection::EdgeDetection(const vole::EdgeDetectionConfig &cfg)
  : config(cfg), vole::Command("edge_detect", // command name
		  config,
		  "Johannes Jordan, Ralph Muessig", // author names
		  "johannes.jordan@cs.fau.de" ) // email
{}

SOM* EdgeDetection::train(const multi_img &img)
{
	SOM *som;
	if (config.graph_withGraph || config.withUMap) {
		som = new GraphSOM(config, img.size());
	} else {
		som = new SOM(config, img.size());
	}
	std::cout << "# Generated SOM " << config.som_width << "x" << config.som_height << " with dimension " << img.size() << std::endl;

	SOMTrainer trainer(*som, img, config);

	std::cout << "# SOM Trainer starts to feed the network using "<< config.som_maxIter << " iterations..." << std::endl;

	vole::Stopwatch watch("Training");
	trainer.feedNetwork();

	return som;
}

int EdgeDetection::execute()
{
	assert(!config.prefix_enabled); // input, output file variables set

/*	if (config.mode.compare("simple"))
		return executeSimple();*/

	multi_img img;
	img.minval = 0.;
	img.maxval = 1.;
	img.read_image(config.input_file);
	if (img.empty())
		return -1;
	img.rebuildPixels(false);

	if (config.hack3d) {
		assert(config.som_height == 1);
		config.som_height = config.som_width * config.som_width;
	}

	SOM *som = train(img);
	if (config.output_som) {
		multi_img somimg = som->export_2d();
		somimg.write_out(config.output_dir + "/som");
		config.storeConfig((config.output_dir + "/config.txt").c_str());
	}

	std::cout << "# Generating 2D image using the SOM and the multispectral image..." << std::endl;
	vole::Stopwatch watch("Edge Image Generation");
	SOMTester tester(*som, img, config);

	if (config.linearization.compare("NONE") == 0)
	{
		std::cout << "# Using direct distance" << std::endl;

		cv::Mat1d dX, dY;
		tester.getEdge(dX, dY);
		std::cout << "Write images" <<std::endl;

		cv::Mat sobelXShow, sobelYShow;

		dX.convertTo(sobelXShow, CV_8UC1, 255.);
		dY.convertTo(sobelYShow, CV_8UC1, 255.);
		std::string xname, yname;	// TODO bullshit
		if (config.graph_withGraph) {
			xname = "/graphEdgeX";
			yname = "/graphEdgeY";
		} else {
			xname = "/directEdgeX";
			yname = "/directEdgeY";
		}

		cv::imwrite(config.output_dir + xname + ".png", sobelXShow);
		cv::imwrite(config.output_dir + yname + ".png", sobelYShow);
	} else if (config.linearization.compare("SFC") == 0) {
		if (config.som_height == 1) {
			std::cout << "# Generating 1D Rank" << std::endl;
			tester.generateRankImage();
		} else {
			if (config.som_height % 2 == 0) {
				int order = 0;
				int length = 2;
				for (order = 1; order < 10; order++) {
					if (length == config.som_height)
						break;
					length *= 2;
				}
				SpaceFillingCurve curve(SpaceFillingCurve::HILBERT, order);
				std::cout << "# Generating 2D Hilbert Rank" << std::endl;
				tester.generateRankImage(curve.getRankMatrix());
			} else if( config.som_height % 3 == 0 ) {
				int order = 0;
				int length = 3;
				for (order = 1; order < 10; order++) {
					if (length == config.som_height)
						break;
					length *= 3;
				}
				SpaceFillingCurve curve(SpaceFillingCurve::PEANO, order);
				std::cout << "# Generating 1D Peano" << std::endl;
				tester.generateRankImage(curve.getRankMatrix());
			} else { // TODO: what about the width? and the modulo test is *WRONG*
				std::cerr << "Height of SOM must be 2^n, 3^n, n >= 0." << std::cout;
				return 1;
			}
		}
		tester.generateEdgeImage( 5., 20.);
	}
    
	delete som;
	return 0;
}

int EdgeDetection::executeSimple()	// simple method for comparison
{
	std::cout << "### Simple method (no SOM) ###" << std::endl;
	multi_img img(config.input_file);

	if(img.empty())
	{
		std::cout << "Error loading image " <<std::endl;
		return 1;
	}
	img.rebuildPixels(false);

	cv::Mat_<uchar> grayscale(img.height, img.width);
	cv::Mat_<uchar> edge(img.height, img.width);

		for(int i = 0; i < img.height; i++) {
			for( int j = 0; j < img.width; j++) {
			std::vector<multi_img::Value> vec = img(i,j);
			double mean = 0.;
			for(unsigned int k = 0; k < vec.size(); k++)
				mean += vec[k]/(double)vec.size();

			grayscale(i,j) = static_cast<uchar>(mean);
		}
	}

// remove .txt appendix
  std::string name = config.input_file.substr(0,config.input_file.size()-4); // TODO error prone

	cv::Canny( grayscale, edge, 15., 40., 3, true );
	cv::imwrite(config.output_dir+name+"_avg.png", grayscale);
	cv::imwrite(config.output_dir+name+"_avg_edge.png", edge);
	std::cout << "# Averaged edge image written using " << img.size() << " bands" << std::endl;

	return 0;
}


void EdgeDetection::printShortHelp() const 
{
	std::cout << "Edge detection in multispectral images using SOM." << std::endl;
}

void EdgeDetection::printHelp() const 
{
	// TODO
}
