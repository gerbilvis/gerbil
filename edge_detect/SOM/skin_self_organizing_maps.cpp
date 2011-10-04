#include <iostream>
#include "skin_self_organizing_maps.h"
#include "defines.h"

// OpenCV
#include "cv.h"

// for imread, imshow etc.
#include "highgui.h"

// we could additionally use namespace cv, but personally I prefer to see where
// my types come from, so I'll type the four letters ("cv::") always in front
// of opencv commands
using namespace boost::program_options;

// in the constructor we set up all parameters the user may configure. Note
// that the command name may deviate from the class name.
skinSelfOrganizingMaps::skinSelfOrganizingMaps()
 : Command(
		"skin_self_organizing_maps", // command name
		"Christoph Malskies", // author(s)
		"chriss@da4ever.de") // email
{
	// here we fill in options the user may set
	// note that every option appears in a bracket pair after the add_options()
	// call.
	// The first parameter is the option title (as to be entered by the user),
	// the second parameter is where to store it + a pre-initialization, the
	// third parameter is a short description that is displayed with the help.
	options.add_options()
		("skin_self_organizing_maps.mode", value(&config.uiMode)->default_value(0),
				"Execution mode: 0=classification 1=training")
		("skin_self_organizing_maps.database", value(&config.cpDataBase)->default_value(""),
		                "Path to Image Database")
		("skin_self_organizing_maps.skin_pixels", value(&config.uiSkinPixels)->default_value(300000),
		                "Number of skin pixels to load from database")
		("skin_self_organizing_maps.non_skin_pixels", value(&config.uiNonSkinPixels)->default_value(0),
				"Number of nonskin pixels to load from database")
		("skin_self_organizing_maps.pixel_per_file", value(&config.uiPixelPerFile)->default_value(200),
				"Number of pixels loaded per file")
		("skin_self_organizing_maps.iterations", value(&config.uiIterations)->default_value(100000),
				"Number of training iterations")
		("skin_self_organizing_maps.calibrations", value(&config.uiCalibrations)->default_value(60000),
				"Number of calibration steps")
		("skin_self_organizing_maps.som", value(&config.cpSOMFile)->default_value(""),
				"SOM file for loading/saving")
		("skin_self_organizing_maps.size", value(&config.uiSOMSize)->default_value(64),
				"Size of SOM as NxN")
		("skin_self_organizing_maps.threshold", value(&config.dThreshold)->default_value(0.00316),
				"Threshold for Skin-Only classification")
		("skin_self_organizing_maps.skin_only", value(&config.uiSkinOnly)->default_value(1),
				"1 if you want to use Skin-Only SOM, 0 for Skin-NonSkin SOM")
		("skin_self_organizing_maps.rate_start", value(&config.dLearnRateStart)->default_value(0.1),
				"Learningrate at training start")
		("skin_self_organizing_maps.rate_end", value(&config.dLearnRateEnd)->default_value(0.0001),
				"Learningrate at training end")
		("skin_self_organizing_maps.radius_start", value(&config.dRadiusStart)->default_value(0.0),
				"Adaptionradius at training start (0 set automatically)")
		("skin_self_organizing_maps.radius_end", value(&config.dRadiusEnd)->default_value(4.0),
				"Adaptionradius at trainign end")
		;
}

// the execute method is the starting point for every vole command
int skinSelfOrganizingMaps::execute() {

	std::cout << std::endl << "****** Self Organizing Maps ******" << std::endl << std::endl;

	//check the mode we are workin in
	//mode == 0 => classification
	if (config.uiMode == 0) {
		
		std::cout << "Classification Mode: ";
		if (config.uiSkinOnly != 0) std::cout << "Skin-Only" << std::endl << std::endl << std::endl;
		else std::cout << "Skin-NonSkin" << std::endl << std::endl << std::endl;

		//check if input file given
		if (global_config.inputfile.length() < 1) {
			std::cout << "ERROR: No input file name given. Please specify an input file (\"-I <file>\")" 
				<< std::endl << std::endl;
			return 1;
		}
	
		//check if SOM file given
		if (config.cpSOMFile.length() == 0) {
			std::cout << "ERROR: No SOM file given. Please specify a SOM file (\"--skin_self_organizing_maps.cpSOMFile <file>.som\")"
				<< std::endl << std::endl;
			return 1;
		}
	
		//load input image
		cv::Mat_<cv::Vec3b> img = cv::imread(global_config.inputfile);
		if (!img.data) {
			std::cout << "ERROR loading image " << global_config.inputfile << std::endl << std::endl;
			return -1;
		}
		std::cout << "Loaded image with size: " << img.cols << "x" << img.rows << std::endl;
	
		//load SOM from file
		SelfOrganisingMap* map;
		if (!SelfOrganisingMap::loadFromFile(&map, config.cpSOMFile.c_str())) {
			std::cout << "ERROR loading SOM " << config.cpSOMFile << std::endl << std::endl;
		return -1;
		}
		std::cout << "Loaded SOM " << map->getWidth() << "x" << map->getHeight() << " using Theta=" << config.dThreshold << std::endl;


		//classificate image
		std::cout << "Working on image..." << std::endl;
		int ret = 0;
		if (config.uiSkinOnly != 0) {
			ret = executeSkinOnly(map, &img, config.dThreshold);
		}
		else {
			ret = executeSkinNonSkin(map, &img);
		}

		//perform outputs
		if (global_config.graphical) {
			cv::imshow("skin pixels marked red", img);
			cv::waitKey();
		}

		if (global_config.outputdir.length() > 0) {

			// another issue: since we write png-images, we can set the compression
			// level (see pp. 838 in the documentation).
			// Here, we choose the maximum compression.
			std::vector<int> compression_params;
			compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
			compression_params.push_back(9);
	
			// again, the command looks like the matlab command
			imwrite(std::string(global_config.outputdir + "/skin_self_organizing_maps_out.png"), img, compression_params);

			std::cout << "Image saved to " << std::string(global_config.outputdir + "/skin_self_organizing_maps_out.png") << std::endl;
		}

		delete map;
	}
	//mode != 0 => training
	else {
		std::cout << "Training Mode: ";
		if (config.uiSkinOnly != 0) std::cout << "Skin-Only" << std::endl << std::endl << std::endl;
		else std::cout << "Skin-NonSkin" << std::endl << std::endl << std::endl;

		//check input parameters
		if (config.cpDataBase.length() == 0) {
			std::cout << "ERROR: No database given. Please specify database (\"--skin_self_organizing_maps.database <path>\")" << std::endl << std::endl;
			return 1;
		}
		if (config.cpSOMFile.length() == 0) {
			std::cout << "ERROR: No SOM file given. Please specify SOM file (\"--skin_self_organizing_maps.som <file>.som\")" << std::endl << std::endl;
			return 1;
		}
		if (config.uiSOMSize < 12) {
			std::cout << "ERROR: Invalid map size. Map size must be > 12." << std::endl << std::endl;
			return 1;
		}

		//create SelfOrganisingMap
		std::cout << "Create and initialize SOM" << std::endl;
		std::cout << "Size: " << config.uiSOMSize << "x" << config.uiSOMSize << std::endl;
		SelfOrganisingMap* map = new SelfOrganisingMap(config.uiSOMSize, config.uiSOMSize);
		map->randomize();
		std::cout << "successfull" << std::endl << std::endl;		

		//create Database and Trainer object
		std::cout << "Loading database... " << std::endl;
		std::cout << "Skin Pixels: " << config.uiSkinPixels << std::endl;
		std::cout << "NonSkin Pixels: " << config.uiNonSkinPixels << std::endl;
		std::cout << "Pixel per file: " << config.uiPixelPerFile << std::endl;
		SOMDatabase db;
		if (!db.load(config.cpDataBase.c_str(), config.uiSkinPixels, config.uiNonSkinPixels, config.uiPixelPerFile)) {
			std::cout << "error loading database" << std::endl;
			return -1;
		}
		std::cout << "successfull " << std::endl << std::endl;
		
		//calculate standard adaptionradius if not given
		if (config.dRadiusStart == 0.0) {
			config.dRadiusStart = config.uiSOMSize/3;
		}
		SOMTrainer trainer;
		trainer.setParameters(config.dLearnRateStart, config.dLearnRateEnd, config.dRadiusStart, config.dRadiusEnd, config.uiIterations);	
	
		//training
		std::cout << "SOM Training..." << std::endl;
		std::cout << "Iterations: " << config.uiIterations << std::endl;
		if (config.uiSkinOnly == 0) std::cout << "Calibrations: " << config.uiCalibrations << std::endl;
		std::cout << "Learningrate: " << (float)config.dLearnRateStart << " - " << (float)config.dLearnRateEnd << std::endl;
		std::cout << "Adaptionradius: " << (float)config.dRadiusStart << " - " << (float)config.dRadiusEnd << std::endl;
		int ret = 0;
		if (config.uiSkinOnly != 0) {
			ret = trainingSkinOnly(map, &db, &trainer, config.uiIterations);
		}
		else {
			ret = trainingSkinNonSkin(map, &db, &trainer, config.uiIterations, config.uiCalibrations);
		}
		std::cout << "successfull" << std::endl << std::endl;

		//save the trained SOM to given file
		SelfOrganisingMap::saveToFile(map, config.cpSOMFile.c_str());
		std::cout << "SOM saved to " << config.cpSOMFile << std::endl << std::endl;

		delete map;
		std::cout << "Finished" << std::endl;
	}

	//once we're here everything went fine
    	return 0;
}


void skinSelfOrganizingMaps::printShortHelp() {
	std::cout << "Skin detection with Self Organizing Maps" << std::endl;
}


void skinSelfOrganizingMaps::printHelp() {
	std::cout << "Self Organizing Maps Usage:" << std::endl << std::endl;

	std::cout << "Two modes supported:" << std::endl;
	std::cout << "1. Classification: Skin pixels are marked red (--skin_self_organizing_maps.mode 0)" << std::endl;
	std::cout << "2. Training: Train SOM and store it to a file (--skin_self_organizing_maps.mode 1)" << std::endl;
	std::cout << "Both modes can be used with Skin-Only and Skin-NonSkin SOMs (--skin_self_organizing_maps.skin_only 0/1)" << std::endl << std::endl;

	std::cout << "vole skin_self_organizing_maps -I <image> -O <output_dir> --skin_self_organizing_maps.som <your_som>.som" << std::endl;
	std::cout << "vole skin_self_organizing_maps --skin_self_organizing_maps.mode 1 --skin_self_orgainizing_maps.database <path> --skin_self_organizing_maps.som <your_som>.som" << std::endl << std::endl;

	std::cout << "Image Database must be of following form:" << std::endl;
	std::cout << "masks (folder containing mask images)" << std::endl;
	std::cout << "non-skin-images (folder containing NonSkin Images)" << std::endl;
	std::cout << "skin-images (folder containing Skin Images)" << std::endl;
	std::cout << "filenames_masks.txt (File containing mask image filenames; One file per line)" << std::endl;
	std::cout << "filenames_nonskin.txt (File containing nonskin image filenames; One file per line)" << std::endl;
	std::cout << "filenames_skin.txt (File containing skin image filenames; One file per line)" << std::endl;
}

int skinSelfOrganizingMaps::executeSkinOnly(SelfOrganisingMap* map, cv::Mat_<cv::Vec3b>* img, double threshold) {
	
	//calculate squared threshold (we can save the squareroot)
	double sqThreshold = threshold*threshold;

	//go through image
	for (int y = 0; y < img->rows; ++y) {
		cv::Vec3b* current_row = (*img)[y];
		for (int x = 0; x < img->cols; ++x) {

			//calculate normalised RG
			unsigned char r = current_row[x][2];
			unsigned char g = current_row[x][1];
			unsigned char b = current_row[x][0];

			double n = 0.0;
			double c1 = 0.0;
			double c2 = 0.0;

			n = (double)r + (double)g + (double)b;
			if (n != 0.0) {
				c1 = (double)r/n;
				c2 = (double)g/n;
			}

			//get winner neuron and mark pixel as skin if distance(squared) is smaller then threhsold(squared)
			Neuron* win = map->getWinner(c1, c2, NULL);
			double distance = (c1-win->c1)*(c1-win->c1) + (c2-win->c2)*(c2-win->c2);
			if (distance < 	sqThreshold) {
				current_row[x][2] = 255;
				current_row[x][1] = 0;
				current_row[x][0] = 0;
			}
		}
	}
	return 0;
}

int skinSelfOrganizingMaps::executeSkinNonSkin(SelfOrganisingMap* map, cv::Mat_<cv::Vec3b>* img) {
	
	//go through image
	for (int y = 0; y < img->rows; ++y) {
		cv::Vec3b* current_row = (*img)[y];
		for (int x = 0; x < img->cols; ++x) {
			
			//calculate normalised RG
			unsigned char r = current_row[x][2];
			unsigned char g = current_row[x][1];
			unsigned char b = current_row[x][0];

			double n = 0.0;
			double c1 = 0.0;
			double c2 = 0.0;

			n = (double)r+(double)g+(double)b;
			if (n != 0.0) {
				c1 = (double)r/n;
				c2 = (double)g/n;
			}

			//get winner neuron and assign red for skin pixel if winner has positive label
			Neuron* win = map->getWinner(c1, c2, NULL);
			if (win->l > 0) {
				current_row[x][2] = 255;
				current_row[x][1] = 0;
				current_row[x][0] = 0;
			}
		}
	}
	return 0;
}

int skinSelfOrganizingMaps::trainingSkinOnly(SelfOrganisingMap* map, SOMDatabase* db, SOMTrainer* trainer, unsigned int uiIterations) {
	
	Color c;

	//training
	for (unsigned int i = 0; i < uiIterations; ++i) {
		//get skin color value from databse
		db->getSkin(rand()%db->numSkinPixels(), c);
		
		unsigned char r = c.R;
		unsigned char g = c.G;
		unsigned char b = c.B;

		double n = 0.0;
		double c1 = 0.0;
		double c2 = 0.0;

		n = (double)r+(double)g+double(b);
		if (n != 0.0) {
			c1 = (double)r/n;
			c2 = (double)g/n;
		}
		
		//perform one iteration step
		trainer->iterate(*map, c1, c2);
	}

	return 0;
}

int skinSelfOrganizingMaps::trainingSkinNonSkin(SelfOrganisingMap* map, SOMDatabase* db, SOMTrainer* trainer, unsigned int uiIterations, unsigned int uiCalibrations) {
	
	Color c;
	Neuron* win;

	//training
	for (unsigned int i = 0; i < uiIterations; ++i) {
		//get color value from database, alternating skin <=> nonskin
		if (i%2 == 0) db->getSkin(rand()%db->numSkinPixels(), c);
		else db->getNonSkin(rand()%db->numNonSkinPixels(), c);
		
		//convert to normalised RG
		unsigned char r = c.R;
		unsigned char g = c.G;
		unsigned char b = c.B;

		double n = 0.0;
		double c1 = 0.0;
		double c2 = 0.0;

		n = (double)r+(double)g+double(b);
		if (n != 0.0) {
			c1 = (double)r/n;
			c2 = (double)g/n;
		}
		
		//perform one iteration step
		trainer->iterate(*map, c1, c2);
	}

	//calibration
	for (unsigned int i = 0; i < uiCalibrations; ++i) {
		//get color value from database, alternating skin <=> nonskin
		if (i%2 == 0) db->getSkin(rand()%db->numSkinPixels(), c);
		else db->getNonSkin(rand()%db->numNonSkinPixels(), c);
		
		//convert to normalised RG
		unsigned char r = c.R;
		unsigned char g = c.G;
		unsigned char b = c.B;

		double n = 0.0;
		double c1 = 0.0;
		double c2 = 0.0;

		n = (double)r+(double)g+double(b);
		if (n != 0.0) {
			c1 = (double)r/n;
			c2 = (double)g/n;
		}

		//adjust winner label
		win = map->getWinner(c1, c2, NULL);
		if (i%2 == 0) win->l += 1;
		else win->l -= 1;
	}

	return 0;
}

