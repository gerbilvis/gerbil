#ifndef SOM_CONFIG_H
#define SOM_CONFIG_H

#include <vole_config.h>
#include <sm_config.h>

namespace som {

// SomType
enum Type {
	// n-dimensional isometric
	SOM_SQUARE = 0x1,
	SOM_CUBE = 0x2,
	SOM_TESSERACT = 0x4
};
// note: the string works with values 0, 1, 2.. so no gaps!
#define som_TypeString {"none", "square", "cube", "tesseract"}

class SOMConfig : public Config {

public:
	SOMConfig(const std::string& prefix = std::string());

	virtual ~SOMConfig() {}

	Type type;

	// size of the dimensions
	int dsize;

	// random seed
	uint64 seed;
	
    //bool use_opencl;
    //bool use_opencl_cpu_opt;

	// Training features 
	int maxIter;		// number of iterations
	double learnStart;	// start value for learning rate (fades off with sigma)
	double learnEnd;	// start value for learning rate (fades off with sigma)
	double sigmaStart;	// start value for neighborhood radius
	double sigmaEnd;	// start value for neighborhood radius

	// kernel type: uniform or gauss
	bool gaussKernel;

	// TODO: add bool flag, to explicitly allow overwriting if file exists.
	std::string somFile;

	/// similarity measure for model vector search in SOM
	similarity_measures::SMConfig similarity;

	virtual std::string getString() const;

protected:

#ifdef WITH_BOOST
	virtual void initBoostOptions();
#endif // WITH_BOOST
};

}
#endif
