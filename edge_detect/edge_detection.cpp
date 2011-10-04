//	-------------------------------------------------------------------------------------------------------------	 	//
// 														Variants of Self-Organizing Maps																											//
// Studienarbeit in Computer Vision at the Chair of Patter Recognition Friedrich-Alexander Universitaet Erlangen		//
// Start:	15.11.2010																																																//
// End	:	16.05.2011																																																//
// 																																																									//
// Ralph Muessig																																																		//
// ralph.muessig@e-technik.stud.uni-erlangen.de																																			//
// Informations- und Kommunikationstechnik																																					//
//	---------------------------------------------------------------------------------------------------------------	//



//system includes
#include <iostream>
#include <sstream>

#include <fstream>

//cv includes
#include <cv.h>
#include <highgui.h>

//vole includes
#include <stopwatch.h>
#include <talkingwatch.h>

//som includes
#include "edge_detection.h"
#include "self_organizing_map.h"
#include "som_trainer.h"
#include "GTM/gtm.h"
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
	std::cout << config.algorithm <<std::endl; 
	
	if(config.algorithm == "GTM")
	{	
		std::cout << "# Loading: " << (config.input_dir+config.msi_name)<<std::endl;
		multi_img img;
		img.minval = 0.;
		img.maxval = 1.;
		img.read_image(config.input_dir+config.msi_name);
		img.rebuildPixels(false);
		if (img.empty())
			return -1;
    
		std::cout << "### Application for edge detection in multispectral images ###" << std::endl;
		std::cout << "### Using variants of self-organizing-maps ###" << std::endl;
		std::cout << "# Loaded multispectral image '" << config.msi_name << "' from: " << config.input_dir << std::endl;

		GTM gtm(&config, &img);
		gtm.execute();

		return 0;
	}	
    
	std::fstream out; 
	std::cout << "### Application for edge detection in multispectral images ###" << std::endl;
	std::cout << "### Using variants of self-organizing-maps ###" << std::endl;
  // check some parameters.....
	if (config.output_dir.size() < 1) 
	{
		std::cerr << "Specify output directory by typing -O <output_dir>" << std::endl;
		return 1;
	}

	if (config.input_dir.size() < 1) 
	{
		std::cerr << "please add -I <input_file>" << std::endl;
	return 1;
	}
  
	int m_order = 1;

	if( config.som_height % 2 == 0) 
	{
		int order = 0;
		int length = 2;
		
		for(order = 1; order < 10; order++) 
		{
			if(length == config.som_height) 
				break;
			length *= 2;
		}
    
    m_order = order;
		
  }
	else if( config.som_height % 3 == 0 ) 
	{
		int order = 0;
		int length = 3;
		for(order = 1; order < 10; order++) 
		{
			if(length == config.som_height)
				break;
			length *= 3;
		}
		m_order = order;
	}

// Let the SOM learn and adjust their neurons from the given msi-data
	if (config.mode.compare("learn") == 0 )  
	{
		multi_img img;
		img.minval = 0.;
		img.maxval = 1.;
		std::cout << "# Loading: " << (config.input_dir+config.msi_name)<<std::endl;
		img.read_image(config.input_dir+config.msi_name);
		if (img.empty())
			return -1;

		img.rebuildPixels(false);
		
		std::cout << "# Loading: " << (config.input_dir+config.msi_name)<<std::endl;
    
		vole::Talkingwatch w_overall;
		std::stringstream compTime;
    
		std::cout << "# Loaded multispectral image '" << config.msi_name << "' from: " << config.input_dir << std::endl;
    
		vole::Talkingwatch w_training;//also graphs are initialized here!
		SelfOrganizingMap som(&config, img.size());
		std::cout << "# Generated SOM " << config.som_width << "x" << config.som_height << " with dimension " << img.size() << std::endl;

		SomTrainer trainer(som, img, config, config.msi_name);

		std::cout << "# SOM Trainer starts to feed the network using "<< config.som_maxIter << " iterations..." << std::endl;
    
		trainer.feedNetwork();
		compTime << w_training.print("Training");
		std::cout << w_training.print("Training");

		std::cout << "# Generating 2D image using the SOM and the multispectral image..." << std::endl;
    
		if (config.linearization.compare("NONE") == 0)
		{
			std::cout << "# Using direct distance" << std::endl;

			vole::Talkingwatch w_edge;

			cv::Mat1d dX, dY;
			
			trainer.getEdge(dX,dY, 0);
			compTime << w_edge.print("Edge image generation");
			std::cout << w_edge.print("Edge image generation");
			std::cout << "Write images" <<std::endl;
			
			cv::Mat sobelXShow,sobelYShow;

			dX.convertTo( sobelXShow, CV_8UC1, 255.);
			dY.convertTo( sobelYShow, CV_8UC1, 255.);
			std::string xname, yname;
			std::string umap ="";
			if(config.withUMap)
				umap = "";
			if(config.graph_withGraph)
			{	
				if(config.sw_phi > 0.0 || config.sw_beta > 0.0)
				{	
					xname = trainer.getFilenameExtension() + umap +"_graphEdgeX";
					yname = trainer.getFilenameExtension() + umap + "_graphEdgeY";
				}
				else
				{
					xname = trainer.getFilenameExtension() + umap +"_graphEdgeX";
					yname = trainer.getFilenameExtension() + umap +"_graphEdgeY";
				}	
			}
			else
			{
				xname = trainer.getFilenameExtension() + umap +"_directEdgeX";
				yname = trainer.getFilenameExtension() + umap +"_directEdgeY";
			}		
			
			compTime << w_overall.print("Complete Time");
			std::cout <<w_overall.print("Complete Time");
			std::cout << "Done" <<std::endl;

			std::ofstream outFile;
			std::string name = config.output_dir + xname + "_time_NONE.txt";
			outFile.open(name.c_str(),std::ios_base::out | std::ios_base::trunc);
			if(!outFile.is_open())
				std::cout << "File not open!" <<std::endl;
			outFile << compTime.str();
			xname += ".png"; 
			yname += ".png";
			outFile << trainer.graphProperties();
			outFile.close();
			cv::imwrite(config.output_dir+ xname, sobelXShow);
			cv::imwrite(config.output_dir+ yname, sobelYShow);
      
		}
    else if (config.linearization.compare("SFC") == 0)
		{
			vole::Talkingwatch w_edge;
 
			if(config.som_height == 1 ) 
			{
				std::cout << "# Generating 1D Rank" << std::endl;
				trainer.generateRankImage();
			}
			else 
			{
				if( config.som_height % 2 == 0) 
				{
					SpaceFillingCurve curve(SpaceFillingCurve::HILBERT, m_order);
					//curve.visualizeMatrix();
					std::cout << "# Generating 2D Hilbert Rank" << std::endl;
					trainer.generateRankImage(curve.getRankMatrix());

				} 
				else if( config.som_height % 3 == 0 ) 
				{
					SpaceFillingCurve curve(SpaceFillingCurve::PEANO, m_order);
					//curve.visualizeMatrix();
					std::cout << "# Generating 1D Peano" << std::endl;
					trainer.generateRankImage(curve.getRankMatrix());

				}
				else 
				{
					std::cerr << "Height of SOM must be 2^n, 3^n, n >= 0." << std::cout;
					return 1;
				}
      }
			
			trainer.generateEdgeImage( 5., 20.);
			compTime << w_edge.print("Edge Image Generation");
			compTime << w_overall.print("Complete Time");
		
			std::ofstream outFile;
			std::string name = config.output_dir + trainer.getFilenameExtension() + "_time_SCF.txt";
			outFile.open(name.c_str(),std::ios_base::out | std::ios_base::trunc);
			if(!outFile.is_open())
				std::cout << "File not open!" <<std::endl;
			outFile << compTime.str();
			outFile.close();
		}
    
		return 0;
	}

  // Simple edge detection using averaged pixel vectors and Canny
  if (config.mode.compare("simpleEdges") == 0 ) 
	{
		multi_img img(config.input_dir+config.msi_name);;

		if(img.empty())
    {
			std::cout << "# Error loading image " <<std::endl;
			return 1;
		}  
    
		img.rebuildPixels(false);
    
		cv::Mat_<uchar> grayscale(img.height, img.width);
		cv::Mat_<uchar> edge(img.height, img.width);

		for(int i = 0; i < img.height; i++) 
		{
			for( int j = 0; j < img.width; j++) 
			{
				std::vector<multi_img::Value> vec = img(i,j);
				double mean = 0.;
				for(unsigned int k = 0; k < vec.size(); k++)
					mean += vec[k]/(double)vec.size();

				grayscale(i,j) = static_cast<uchar>(mean);
			}
		}

    // remove .txt appendix
		std::string name = config.msi_name.substr(0,config.msi_name.size()-4);;
      
		cv::Canny( grayscale, edge, 15., 40., 3, true );
		cv::imwrite(config.output_dir+name+"_avg.png", grayscale);
		cv::imwrite(config.output_dir+name+"_avg_edge.png", edge);
		std::cout << "# Averaged edge image written using " << img.size() << " bands" << std::endl;

		return 0;
	}

	std::cerr << "# Please specify an operating modus using: -M <learn> or <apply> or <visualize>" << std::endl;
  

	return 0;
}


void EdgeDetection::printShortHelp() const 
{
	std::cout << "Module for edge detection in multispectral images using SOM." << std::endl;
}

void EdgeDetection::printHelp() const 
{
	std::cout << "Module for edge detection in multispectral images" << std::endl;
	std::cout << " using SelfOrganizingMaps." << std::endl;
}
