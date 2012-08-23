#ifndef PROBSHIFT_CONFIG_H
#define PROBSHIFT_CONFIG_H

#include "vole_config.h"
#include "imginput.h"

namespace vole {

/**
 * Configuration parameters
 */
class ProbShiftConfig : public Config {
public:

	ProbShiftConfig(const std::string& prefix = std::string());

	// input configuration
	ImgInputConfig inputconfig;

	// output directory
	std::string output_directory;

	// LSH parameters
	bool useLSH;
	int lshK;
	int lshL;

	// use self-tuning spectral clustering for cluster identification
	bool useSpectral;

	// spectral clustering parameters
	int maxClusts;
	int minClusts;

	// use converged probabilities for spectral clustering
	bool useConverged;

	// do Meanshift post-processing
	bool useMeanShift;

	// scale bandwidths to get a specific cluster count
	int msClusts;

	// mean-shift post-processing bandwidth factor
    double msBwFactor;

	// jump directly to Meanshift post-processing by loading previously saved modes
	std::string loadModes;
	std::string saveModes;

	virtual std::string getString() const;

	virtual ~ProbShiftConfig();

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST
};

}

#endif // PROBSHIFT_CONFIG_H
