#ifndef EDGE_DETECTION_CONFIG_H
#define EDGE_DETECTION_CONFIG_H

#include <vole_config.h>
#include <sm_config.h>
#include <multi_img.h>

namespace vole {

class EdgeDetectionConfig : public Config {

public:
	EdgeDetectionConfig(const std::string& prefix = std::string());

	virtual ~EdgeDetectionConfig() {}

	// input image filename
	std::string input_file;
	// working directory
	std::string output_dir;

	// som file input
	std::string som_file;

	// random seed
	uint64 seed;
	
	// SOM features
	int width;
	int height;
	bool hack3d;

	// export SOM?
	bool output_som;

	// Training features 
	int maxIter;								// number of iterations
	double learnStart;							// start value for learning rate
	double learnEnd;							// start value for learning rate
	double radiusStart;							// start value for neighborhood radius
	double radiusEnd;							// start value for neighborhood radius

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
