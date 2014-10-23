#ifndef RGB_CONFIG_H
#define RGB_CONFIG_H

#include <vole_config.h>
#include <imginput_config.h>
#ifdef WITH_SOM
#include <som_config.h>
#endif

namespace rgb {

enum algorithm {
	COLOR_XYZ,
	COLOR_PCA,
	COLOR_SOM
};
#define rgb_algorithmString {"XYZ", "PCA", "SOM"}

/**
 * Configuration parameters for the graph cut / power watershed segmentation
 */
class RGBConfig : public Config {

public:
	RGBConfig(const std::string& prefix = std::string());

	virtual ~RGBConfig() {}

	// input configuration
	imginput::ImgInputConfig input;

	/// output file name
	std::string output_file;

	/// algorithm to be employed
	algorithm algo;
	
	/// maximize PCA contrast
	bool pca_stretch;
	
	/// number of BMUs to query in SOM case
	int som_depth;
	
	/// linear mixing instead of power-of-two weighting scheme
	bool som_linear;

#ifdef WITH_SOM
	som::SOMConfig som;
#endif

	virtual std::string getString() const;

protected:
	#ifdef WITH_BOOST_PROGRAM_OPTIONS
		virtual void initBoostOptions();
	#endif // WITH_BOOST_PROGRAM_OPTIONS
};

}

#endif
