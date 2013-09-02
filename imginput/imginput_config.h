#ifndef IMGINPUT_CONFIG_H
#define IMGINPUT_CONFIG_H

#include "vole_config.h"

namespace vole {

/**
 * Configuration parameters
 */
class ImgInputConfig : public Config {
public:

	ImgInputConfig(const std::string& prefix = std::string());

	// input file name
	std::string file;
	
	// region of interest
	std::string roi;

	// compute spectral gradient
	bool gradient;

	// reduce number of bands (0 means disabled)
	int bands;

	// Band low for cropping
	int bandlow;

	// Band high for cropping
	int bandhigh;

	virtual std::string getString() const;

	virtual ~ImgInputConfig();

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST
};

}

#endif // IMGINPUT_CONFIG_H
