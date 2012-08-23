#ifndef MEANSHIFT_CONFIG_H
#define MEANSHIFT_CONFIG_H

#include "vole_config.h"
#include <imginput.h>

namespace vole {

enum ms_sampling {
	ALL,
	JUMP,
	PERCENT
};
#define ms_samplingString {"ALL", "JUMP", "PERCENT"}

/**
 * Configuration parameters for the graph cut / power watershed segmentation
 */
class MeanShiftConfig : public Config {
public:

	MeanShiftConfig(const std::string& prefix = std::string());

	// input configuration
	ImgInputConfig inputconfig;

	/// working directory
	std::string output_directory;

	/// write out the total coverage (FALSE) or only label (index) image (TRUE)
	bool batch;

	/// use locality sensitive hashing
	bool use_LSH;
	int K, L; ///<- LSH parameters
	
	/// pilot density
	int k; // number of neighbors used for construction
	
	/// starting points, i.e. sampling
	ms_sampling starting;
	int jump;
	float percent; 
	float bandwidth;
	
	/// find optimal K and L automatically
	bool findKL;
	int Kmin, Kjump;
	float epsilon;

	/// random seed (0 means time-based)
	int seed;

	// minimum number of points per reported mode (after pruning)
	int pruneMinN;
	
	virtual std::string getString() const;

	virtual ~MeanShiftConfig();

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST
};

}

#endif
