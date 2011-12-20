#ifndef EDGE_DETECTION_CONFIG_H
#define EDGE_DETECTION_CONFIG_H

#include <config.h>
#include <sm_config.h>
#include <multi_img.h>

namespace vole {

class EdgeDetectionConfig : public Config {

public:
	EdgeDetectionConfig(const std::string& prefix = std::string());

	virtual ~EdgeDetectionConfig() {}

	// graphical output on runtime?
	bool isGraphical;

	// input image filename
	std::string input_file;
	// working directory
	std::string output_dir;

	// SOM linearization: <NONE> | <SFC>
	std::string linearization;

	// random seed
	uint64 seed;
	
	// SOM features
	int som_width;
	int som_height;
	bool hack3d;

	// export SOM?
	bool output_som;
	// TODO: input SOM functionality

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

	/// similarity measure for model vector search in SOM
	SMConfig similarity;

	virtual std::string getString() const;

	protected:

	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST
};

}

#endif
