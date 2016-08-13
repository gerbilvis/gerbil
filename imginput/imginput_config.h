#ifndef IMGINPUT_CONFIG_H
#define IMGINPUT_CONFIG_H

#include "vole_config.h"

namespace imginput {

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

	// normalize L2
	bool normalize;

	// compute spectral gradient
	bool gradient;

	// reduce number of bands (0 means disabled)
	int bands;

	// Band low for cropping
	int bandlow;

	// Band high for cropping
	int bandhigh;

	// Remove blackbody illuminant with X Kelvin
	int removeIllum;

	// Add blackbody illuminant with X Kelvin
	int addIllum;

	std::string output;

	virtual std::string getString() const;

	virtual ~ImgInputConfig();

protected:
#ifdef WITH_BOOST_PROGRAM_OPTIONS
	virtual void initBoostOptions();
#endif // WITH_BOOST_PROGRAM_OPTIONS
};

}

#endif // IMGINPUT_CONFIG_H
