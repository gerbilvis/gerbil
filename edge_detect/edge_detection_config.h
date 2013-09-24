#ifndef EDGE_DETECTION_CONFIG_H
#define EDGE_DETECTION_CONFIG_H

#include <vole_config.h>
#include <sm_config.h>
#include <multi_img.h>

namespace vole {

enum somtype {
	SOM_LINE,
	SOM_SQUARE,
	SOM_CUBE,
	SOM_CONE
};
#define somtypeString {"line", "square", "cube", "cone"}

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
	int sidelength;		// line, square, cube
	double granularity;	// cone
	somtype type;

	// export SOM?
	bool output_som;
    bool use_opencl;
    bool use_opencl_new;
    bool opencl_test;

	// Training features 
	int maxIter;								// number of iterations
	double learnStart;							// start value for learning rate (fades off with sigma)
	double learnEnd;							// start value for learning rate (fades off with sigma)
	double sigmaStart;							// start value for neighborhood radius
	double sigmaEnd;							// start value for neighborhood radius

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
