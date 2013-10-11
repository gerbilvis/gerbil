#ifndef MEANSHIFT_CONFIG_H
#define MEANSHIFT_CONFIG_H

#include "vole_config.h"
#ifdef WITH_SEG_FELZENSZWALB
#include <felzenszwalb2_config.h>
#endif
#include <imginput.h>

namespace vole {

enum ms_sampling {
	ALL,
	JUMP,
	PERCENT,
#ifdef WITH_SEG_FELZENSZWALB
	SUPERPIXEL,
#endif
};
#ifdef WITH_SEG_FELZENSZWALB
#define ms_samplingString {"ALL", "JUMP", "PERCENT", "SUPERPIXEL"}
#else
#define ms_samplingString {"ALL", "JUMP", "PERCENT"}
#endif

/**
 * Configuration parameters for the graph cut / power watershed segmentation
 */
class MeanShiftConfig : public Config {
public:

	MeanShiftConfig(const std::string& prefix = std::string());

	// input configuration
	ImgInputConfig input;

#ifdef WITH_SEG_FELZENSZWALB
	// superpixel configuration for SUPERPIXEL sampling
	gerbil::FelzenszwalbConfig superpixel;

	// compute superpixels on original image, mean shift on spectral gradient
	bool sp_withGrad;
	// how to weight superpixel sizes: 0 do not weight, 1 fixed bandwidths, 2 alter weightdp2
	int sp_weight;
#endif

	/// working directory
	std::string output_directory;

	/// file prefix
	std::string output_prefix;

	/// write out the total coverage (FALSE) or only label (index) image (TRUE)
	bool batch;

	/// use locality sensitive hashing
	bool use_LSH;
	int K, L; ///<- LSH parameters
	
	/// pilot density
	float k; // k * sqrt(N) is number of neighbors used for construction
	
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
