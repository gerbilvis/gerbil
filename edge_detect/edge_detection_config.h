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


#ifndef EDGE_DETECTION_CONFIG_H
#define EDGE_DETECTION_CONFIG_H

#include <iostream>
#include <string>

#ifdef VOLE_GUI
#include <QWidget>
#endif // VOLE_GUI

#include "config.h"

// this is our config struct. a struct is not mandatory, but it is clean!
class EdgeDetectionConfig : public vole::Config {

	public:
	EdgeDetectionConfig(std::string prefix = std::string());

	// graphical output on runtime?
	bool isGraphical;
	// dir where images are stored
	std::string input_dir;
	// working directory
	std::string output_dir;

	// Edge detection algorithm: <SOM> | <GTM>
	std::string algorithm;
	
	// SOM methods: <learn> | <apply> | <visualize>
	std::string mode;
	  
	// SOM linearization: <NONE> | <SFC>
	std::string linearization;

	// MSI name
	std::string msi_name;

	// use fixed seed for random initializations
	bool fixedSeed;
	
	// SOM features
	int som_width;
	int som_height;
	std::string som_file;


	// Training features 
	bool withUMap;											//use unified distance map for calculating distances, can be used by DD,SW,GTM 
	double scaleUDistance; 							//scale factor, how mow of the distance weight is taken into account
	bool forceDD; 											// use direct distance for edge detection on a graph-trained SOM
  bool graph_withGraph; 							//use Graph topology instead of simple SOM
  unsigned int sw_initialDegree; 			// initial degree for creation of small world topologies
  std::string graph_type; 						// graph topology
  std::string sw_model; 							// type of model creating a small world graph
  double sw_beta;											// probability for rewireing an edge using the beta-model
  double sw_phi;											// percentage for rewireing an edge using the phi-model
	int som_maxIter;										// number of iterations
	double som_learnStart;							// start value for learning rate
	double som_learnEnd;								// start value for learning rate
	double som_radiusStart;							// start value for neighborhood radius
	double som_radiusEnd	;							// start value for neighborhood radius
	
	//GTM parameters
	std::string gtm_actfn; 							// activation function for the RBF net : <GAUSSIAN>, <TPS>, <R4LOGR>
	unsigned int gtm_numRbf;						//rbf shape is gtm_numRbf x gtm_numRbf
	unsigned int gtm_numLatent;					//latent shape is gtm_numlatent x gtm_numLatent
	
	unsigned int gtm_numIterations;			//number of EM iterations
	double gtm_samplePercentage;				//percentage of input data used to train EM

	//! Return available parameters
	virtual std::string getString() const;

	#ifdef VOLE_GUI
		virtual QWidget *getConfigWidget();
		virtual void updateValuesFromWidget();
	#endif// VOLE_GUI

	protected:

	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST

	#ifdef VOLE_GUI
	// qt data structures 
	#endif // VOLE_GUI

};


#endif
