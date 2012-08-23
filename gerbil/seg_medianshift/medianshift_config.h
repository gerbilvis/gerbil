#ifndef MEDIANSHIFT_CONFIG_H
#define MEDIANSHIFT_CONFIG_H

#include "vole_config.h"
#include "imginput.h"

namespace vole {

/**
 * Configuration parameters
 */
class MedianShiftConfig : public Config {
public:

	MedianShiftConfig(const std::string& prefix = std::string());

	/// input configuration
	ImgInputConfig inputconfig;
	
	/// output directory
	std::string output_directory;

	/// LSH parameters
	int K, L;

	// neighborhood for adaptive window sizes
	double k;

	// threshold for significant mode detection (negative = disabled)
	double signifThresh;

	/// skip mode propagation step
	bool skipprop;
	
	virtual std::string getString() const;

	virtual ~MedianShiftConfig();

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST
};

}

#endif // MEDIANSHIFT_CONFIG_H
