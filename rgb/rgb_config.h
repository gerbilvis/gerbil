#ifndef RGB_CONFIG_H
#define RGB_CONFIG_H

#include <vole_config.h>
#include <imginput.h>
#ifdef WITH_EDGE_DETECT
#include <edge_detection_config.h>
#endif
#include <multi_img.h>

namespace gerbil {

enum rgbalg {
	COLOR_XYZ,
	COLOR_PCA,
	COLOR_SOM
};
#define rgbalgString {"XYZ", "PCA", "SOM"}

/**
 * Configuration parameters for the graph cut / power watershed segmentation
 */
class RGBConfig : public vole::Config {

public:
	RGBConfig(const std::string& prefix = std::string());

	virtual ~RGBConfig() {}

	// input configuration
	vole::ImgInputConfig input;

	/// output file name
	std::string output_file;

	/// algorithm to be employed
	rgbalg algo;
	
	/// maximize PCA contrast
	bool pca_stretch;
	
	/// number of BMUs to query in SOM case
	int som_depth;
	
	/// linear mixing instead of power-of-two weighting scheme
	bool som_linear;

#ifdef WITH_EDGE_DETECT
	vole::EdgeDetectionConfig som;
#endif

	virtual std::string getString() const;

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST
};

}

#endif
