#ifndef SKIN_SELF_ORGANIZING_MAPS_H
#define SKIN_SELF_ORGANIZING_MAPS_H

#include <iostream>
#include "command.h"

#include "SelfOrganisingMap.h"
#include "SOMTrainer.h"
#include "SOMDatabase.h"

// our class starts here
class skinSelfOrganizingMaps : public Command {
public:
	skinSelfOrganizingMaps();

	int execute();

	void printShortHelp();
	void printHelp();

private:

	// this is our config struct. a struct is not mandatory, but it is clean!
	//struct {
	//	/** an example boolean value */
	//	bool say_hello_world;
	//	/** region size for the skin pixel window */
	//	unsigned int region_size;
	//} config;

	struct {
		/** Mode 0=classifcation 1=training */
		unsigned int uiMode;
		/** path to pixel database */
		std::string cpDataBase;
		/** number of skin pixels */
		unsigned int uiSkinPixels;
		/** number of nonskin pixels */
		unsigned int uiNonSkinPixels;
		/** number of pixels per file */
		unsigned int uiPixelPerFile;
		/** number of iteratios */
		unsigned int uiIterations;
		/** number of calibrations */
		unsigned int uiCalibrations;
		/** SOM file name (load or save) */
		std::string cpSOMFile;
		/** SOM size (quadratic) */
		unsigned int uiSOMSize;
		/** SkinOnly threshold */
		double dThreshold;
		/** True for SkinOnly */
		unsigned int uiSkinOnly;
		/** Learningrate start */
		double dLearnRateStart;
		/** Learningrate end */
		double dLearnRateEnd;
		/** Adaptionradius start */
		double dRadiusStart;
		/** Adaptionradius end */
		double dRadiusEnd;
	} config;

	int executeSkinOnly(SelfOrganisingMap* map, cv::Mat_<cv::Vec3b>* img, double threshold);
	int executeSkinNonSkin(SelfOrganisingMap* map, cv::Mat_<cv::Vec3b>* img);
	int trainingSkinOnly(SelfOrganisingMap* map, SOMDatabase* db, SOMTrainer* trainer, unsigned int uiIterations);
	int trainingSkinNonSkin(SelfOrganisingMap* map, SOMDatabase* db, SOMTrainer* trainer, unsigned int uiIterations, unsigned int uiCalibrations);
};

#endif // SKIN_SELF_ORGANIZING_MAPS_H
