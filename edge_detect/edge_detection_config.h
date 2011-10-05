#ifndef EDGE_DETECTION_CONFIG_H
#define EDGE_DETECTION_CONFIG_H

#include <config.h>
#include <multi_img.h>
#include <similarity_measure.h>
#include <sm_config.h>
#include <iostream>
#include <string>

#ifdef VOLE_GUI
#include <QWidget>
#endif // VOLE_GUI

// this is our config struct. a struct is not mandatory, but it is clean!
class EdgeDetectionConfig : public vole::Config {

	public:
	EdgeDetectionConfig(std::string prefix = std::string());
	~EdgeDetectionConfig();

	// graphical output on runtime?
	bool isGraphical;
	// dir where images are stored
	std::string input_dir;
	// working directory
	std::string output_dir;

	// SOM methods: <learn> | <apply> | <visualize>
	std::string mode;
	  
	// SOM linearization: <NONE> | <SFC>
	std::string linearization;

	// Image
	std::string msi_name;

	// use fixed seed for random initializations TODO specify seed
	bool fixedSeed;
	
	// SOM features
	int som_width;
	int som_height;
	std::string som_file;
	bool hack3d;

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
	double som_radiusEnd;							// start value for neighborhood radius

	/// similarity measure for edge weighting
	vole::SMConfig similarity;
	vole::SimilarityMeasure<multi_img::Value> *distfun;

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
