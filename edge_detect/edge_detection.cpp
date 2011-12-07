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
          "Ralph Muessig", // author names
          "ralph.muessig@e-technik.stud.uni-erlangen.de" ) // email
{
	logOutput = false;
}

int EdgeDetection::execute()
{
	std::cout << "### Application for edge detection in multispectral images ###" << std::endl;
	if (config.mode.compare("simple"))
		return executeSimple();

	std::cout << "### Using variants of self-organizing-maps ###" << std::endl;
  // check some parameters.....
	if (config.output_dir.size() < 1) {
		std::cerr << "Specify output directory by typing -O <output_dir>" << std::endl;
		return 1;
	}

	if (config.input_dir.size() < 1) {
		std::cerr << "please add -I <input_file>" << std::endl;
		return 1;
	}

	multi_img img;
	img.minval = 0.;
	img.maxval = 1.;
	std::cout << "# Loading: " << (config.input_dir+config.msi_name)<<std::endl;
	img.read_image(config.input_dir+config.msi_name);
	if (img.empty())
		return -1;
	img.rebuildPixels(false);

	std::cout << "# Loaded multispectral image '" << config.msi_name << "' from: " << config.input_dir << std::endl;
	SOM *som;
	if (config.graph_withGraph || config.withUMap) {
		som = new GraphSOM(config, img.size());
	} else {
		som = new SOM(config, img.size());
	}
	std::cout << "# Generated SOM " << config.som_width << "x" << config.som_height << " with dimension " << img.size() << std::endl;

	SOMTrainer trainer(*som, img, config);

	std::cout << "# SOM Trainer starts to feed the network using "<< config.som_maxIter << " iterations..." << std::endl;

	vole::Stopwatch watch;
	trainer.feedNetwork();
	watch.print("Training");

	std::cout << "# Generating 2D image using the SOM and the multispectral image..." << std::endl;

	SOMTester tester(*som, img, config);
	if (config.linearization.compare("NONE") == 0)
	{
		std::cout << "# Using direct distance" << std::endl;

		vole::Stopwatch watch;

		cv::Mat1d dX, dY;
		tester.getEdge(dX, dY);
		watch.print("Calculating Edge image");
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

		cv::imwrite(config.output_dir + xname, sobelXShow);
		cv::imwrite(config.output_dir + yname, sobelYShow);
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
    
	watch.print("Edge Image Generation");
	delete som;
	return 0;
}

int EdgeDetection::executeSimple()	// simple method for comparison
{
	multi_img img(config.input_dir+config.msi_name);

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
  std::string name = config.msi_name.substr(0,config.msi_name.size()-4); // TODO error prone

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
